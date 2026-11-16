//! Independent CBLAS argument checks and provider/reference comparisons.
use super::*;
use scioxide_core::{DiagonalKind, Triangle};
use scioxide_linalg::{DirectSolveOptions, MatrixSide, MatrixViewMut, RankPolicy, TriangularPlan};

pub(super) fn validate_call<S: NumericScalar>(call: &BlasCall<'_, S>, caps: &BlasCapabilities) {
    let integer = |value: usize| {
        assert!(
            value
                <= match caps.integer_width() {
                    BlasIntegerWidth::Lp64 => i32::MAX as usize,
                    BlasIntegerWidth::Ilp64 => i64::MAX as usize,
                }
        )
    };
    let check_transform = |op| {
        assert_ne!(op, BlasTransform::Conjugate, "CBLAS matrix transform");
        assert!(caps.supports_transform(op));
    };
    let matrix = |a: BlasMatrix<'_, S>| {
        check_dense(
            a.rows(),
            a.cols(),
            a.leading_dimension(),
            a.layout(),
            a.values().len(),
        );
        assert!(caps.supports_layout(a.layout()));
        for v in [a.rows(), a.cols(), a.leading_dimension()] {
            integer(v);
        }
    };
    let output = |a: &BlasMatrixMut<'_, S>| {
        check_dense(
            a.rows(),
            a.cols(),
            a.leading_dimension(),
            a.layout(),
            a.values().len(),
        );
        assert!(caps.supports_layout(a.layout()));
        for v in [a.rows(), a.cols(), a.leading_dimension()] {
            integer(v);
        }
    };
    assert!(
        caps.operation_ulp(S::kind(), call_operation(call))
            .is_some()
    );
    match call_operation(call) {
        BlasOperation::Symv
        | BlasOperation::Sbmv
        | BlasOperation::Spmv
        | BlasOperation::Syr
        | BlasOperation::Syr2
        | BlasOperation::Spr
        | BlasOperation::Spr2 => assert!(S::IS_REAL),
        BlasOperation::Herk
        | BlasOperation::Hemv
        | BlasOperation::Hbmv
        | BlasOperation::Hpmv
        | BlasOperation::Her
        | BlasOperation::Her2
        | BlasOperation::Hpr
        | BlasOperation::Hpr2
        | BlasOperation::Hemm
        | BlasOperation::Her2k
        | BlasOperation::Gerc => assert!(!S::IS_REAL),
        _ => (),
    }
    match call {
        BlasCall::Copy { x, y, .. } => {
            check_vector(x.len(), x.stride(), x.values().len());
            integer(x.len());
            integer(x.stride());
            assert!(caps.supports_transform(if x.is_conjugated() {
                BlasTransform::Conjugate
            } else {
                BlasTransform::None
            }));
            check_vector(y.len(), y.stride(), y.values().len());
            integer(y.len());
            integer(y.stride());
            assert_eq!(x.len(), y.len());
        }
        BlasCall::Scale { x, .. } => {
            check_vector(x.len(), x.stride(), x.values().len());
            integer(x.len());
            integer(x.stride());
        }
        BlasCall::Axpy { x, y, .. } => {
            check_vector(x.len(), x.stride(), x.values().len());
            integer(x.len());
            integer(x.stride());
            assert!(caps.supports_transform(if x.is_conjugated() {
                BlasTransform::Conjugate
            } else {
                BlasTransform::None
            }));
            check_vector(y.len(), y.stride(), y.values().len());
            integer(y.len());
            integer(y.stride());
            assert_eq!(x.len(), y.len());
        }
        BlasCall::Dotu { x, y, .. } => {
            check_vector(x.len(), x.stride(), x.values().len());
            integer(x.len());
            integer(x.stride());
            assert!(caps.supports_transform(if x.is_conjugated() {
                BlasTransform::Conjugate
            } else {
                BlasTransform::None
            }));
            check_vector(y.len(), y.stride(), y.values().len());
            integer(y.len());
            integer(y.stride());
            assert!(caps.supports_transform(if y.is_conjugated() {
                BlasTransform::Conjugate
            } else {
                BlasTransform::None
            }));
            assert_eq!(x.len(), y.len());
        }
        BlasCall::Dotc { x, y, .. } => {
            check_vector(x.len(), x.stride(), x.values().len());
            integer(x.len());
            integer(x.stride());
            assert!(caps.supports_transform(if x.is_conjugated() {
                BlasTransform::Conjugate
            } else {
                BlasTransform::None
            }));
            check_vector(y.len(), y.stride(), y.values().len());
            integer(y.len());
            integer(y.stride());
            assert!(caps.supports_transform(if y.is_conjugated() {
                BlasTransform::Conjugate
            } else {
                BlasTransform::None
            }));
            assert_eq!(x.len(), y.len());
        }
        BlasCall::Norm2 { x, .. } => {
            check_vector(x.len(), x.stride(), x.values().len());
            integer(x.len());
            integer(x.stride());
            assert!(caps.supports_transform(if x.is_conjugated() {
                BlasTransform::Conjugate
            } else {
                BlasTransform::None
            }));
        }
        BlasCall::Asum { x, .. } => {
            check_vector(x.len(), x.stride(), x.values().len());
            integer(x.len());
            integer(x.stride());
            assert!(caps.supports_transform(if x.is_conjugated() {
                BlasTransform::Conjugate
            } else {
                BlasTransform::None
            }));
        }
        BlasCall::Iamax { x, .. } => {
            check_vector(x.len(), x.stride(), x.values().len());
            integer(x.len());
            integer(x.stride());
            assert!(caps.supports_transform(if x.is_conjugated() {
                BlasTransform::Conjugate
            } else {
                BlasTransform::None
            }));
        }
        BlasCall::Gemv {
            transform, a, x, y, ..
        } => {
            check_transform(*transform);
            matrix(*a);
            check_vector(x.len(), x.stride(), x.values().len());
            integer(x.len());
            integer(x.stride());
            assert!(caps.supports_transform(if x.is_conjugated() {
                BlasTransform::Conjugate
            } else {
                BlasTransform::None
            }));
            check_vector(y.len(), y.stride(), y.values().len());
            integer(y.len());
            integer(y.stride());
            assert_eq!(
                if matches!(transform, BlasTransform::Transpose | BlasTransform::Adjoint) {
                    (a.cols(), a.rows())
                } else {
                    (a.rows(), a.cols())
                },
                (y.len(), x.len())
            );
        }
        BlasCall::Gemm {
            left_transform,
            right_transform,
            a,
            b,
            c,
            ..
        } => {
            check_transform(*left_transform);
            check_transform(*right_transform);
            matrix(*a);
            matrix(*b);
            output(c);
            assert_eq!(a.layout(), c.layout());
            assert_eq!(b.layout(), c.layout());
            let left = matrix_logical_shape(*a, *left_transform);
            let right = matrix_logical_shape(*b, *right_transform);
            assert_eq!(left.1, right.0);
            assert_eq!((left.0, right.1), (c.rows(), c.cols()));
        }
        BlasCall::Syrk {
            transform, a, c, ..
        } => {
            check_transform(*transform);
            matrix(*a);
            output(c);
            assert_eq!(a.layout(), c.layout());
            assert_eq!(c.rows(), c.cols());
            assert_eq!(matrix_logical_shape(*a, *transform).0, c.rows());
            assert!(matches!(
                transform,
                BlasTransform::None | BlasTransform::Transpose
            ));
        }
        BlasCall::Herk {
            transform, a, c, ..
        } => {
            check_transform(*transform);
            matrix(*a);
            output(c);
            assert_eq!(a.layout(), c.layout());
            assert_eq!(c.rows(), c.cols());
            assert_eq!(matrix_logical_shape(*a, *transform).0, c.rows());
            assert!(matches!(
                transform,
                BlasTransform::None | BlasTransform::Adjoint
            ));
        }
        BlasCall::Trmm {
            transform,
            a,
            b,
            side,
            ..
        } => {
            check_transform(*transform);
            matrix(*a);
            output(b);
            assert_eq!(a.layout(), b.layout());
            assert_eq!(a.rows(), a.cols());
            assert_eq!(
                a.rows(),
                if *side == MatrixSide::Left {
                    b.rows()
                } else {
                    b.cols()
                }
            );
        }
        BlasCall::Trsm {
            transform,
            a,
            b,
            side,
            ..
        } => {
            check_transform(*transform);
            matrix(*a);
            output(b);
            assert_eq!(a.layout(), b.layout());
            assert_eq!(a.rows(), a.cols());
            assert_eq!(
                a.rows(),
                if *side == MatrixSide::Left {
                    b.rows()
                } else {
                    b.cols()
                }
            );
        }
        BlasCall::Swap { x, y, .. } => {
            check_vector(x.len(), x.stride(), x.values().len());
            integer(x.len());
            integer(x.stride());
            check_vector(y.len(), y.stride(), y.values().len());
            integer(y.len());
            integer(y.stride());
            assert_eq!(x.len(), y.len());
        }
        BlasCall::Rot { x, y, .. } => {
            check_vector(x.len(), x.stride(), x.values().len());
            integer(x.len());
            integer(x.stride());
            check_vector(y.len(), y.stride(), y.values().len());
            integer(y.len());
            integer(y.stride());
            assert_eq!(x.len(), y.len());
            assert!(S::IS_REAL);
        }
        BlasCall::Rotm { x, y, .. } => {
            check_vector(x.len(), x.stride(), x.values().len());
            integer(x.len());
            integer(x.stride());
            check_vector(y.len(), y.stride(), y.values().len());
            integer(y.len());
            integer(y.stride());
            assert_eq!(x.len(), y.len());
            assert!(S::IS_REAL);
        }
        BlasCall::Rotg { .. } => {
            assert!(S::IS_REAL);
        }
        BlasCall::Rotmg { .. } => {
            assert!(S::IS_REAL);
        }
        BlasCall::WideDot { x, y, .. } => {
            check_vector(x.len(), x.stride(), x.values().len());
            integer(x.len());
            integer(x.stride());
            assert!(caps.supports_transform(if x.is_conjugated() {
                BlasTransform::Conjugate
            } else {
                BlasTransform::None
            }));
            check_vector(y.len(), y.stride(), y.values().len());
            integer(y.len());
            integer(y.stride());
            assert!(caps.supports_transform(if y.is_conjugated() {
                BlasTransform::Conjugate
            } else {
                BlasTransform::None
            }));
            assert_eq!(x.len(), y.len());
            assert_eq!(S::kind(), ScalarKind::F32);
        }
        BlasCall::OffsetDot { x, y, .. } => {
            check_vector(x.len(), x.stride(), x.values().len());
            integer(x.len());
            integer(x.stride());
            assert!(caps.supports_transform(if x.is_conjugated() {
                BlasTransform::Conjugate
            } else {
                BlasTransform::None
            }));
            check_vector(y.len(), y.stride(), y.values().len());
            integer(y.len());
            integer(y.stride());
            assert!(caps.supports_transform(if y.is_conjugated() {
                BlasTransform::Conjugate
            } else {
                BlasTransform::None
            }));
            assert_eq!(x.len(), y.len());
            assert_eq!(S::kind(), ScalarKind::F32);
        }
        BlasCall::Gbmv {
            transform, a, x, y, ..
        } => {
            check_transform(*transform);
            assert!(caps.supports_layout(BlasLayout::ColumnMajor));
            assert!(a.leading_dimension() > a.lower() + a.upper());
            for n in [
                a.rows(),
                a.cols(),
                a.lower(),
                a.upper(),
                a.leading_dimension(),
            ] {
                integer(n);
            }
            for col in 0..a.cols() {
                for row in col.saturating_sub(a.upper())
                    ..a.rows()
                        .min(col.saturating_add(a.lower()).saturating_add(1))
                {
                    assert!(col * a.leading_dimension() + a.upper() + row - col < a.values().len());
                }
            }
            check_vector(x.len(), x.stride(), x.values().len());
            integer(x.len());
            integer(x.stride());
            assert!(caps.supports_transform(if x.is_conjugated() {
                BlasTransform::Conjugate
            } else {
                BlasTransform::None
            }));
            check_vector(y.len(), y.stride(), y.values().len());
            integer(y.len());
            integer(y.stride());
            assert_eq!(
                if matches!(transform, BlasTransform::Transpose | BlasTransform::Adjoint) {
                    (a.cols(), a.rows())
                } else {
                    (a.rows(), a.cols())
                },
                (y.len(), x.len())
            );
        }
        BlasCall::Symv { a, x, y, .. } => {
            matrix(*a);
            check_vector(x.len(), x.stride(), x.values().len());
            integer(x.len());
            integer(x.stride());
            assert!(caps.supports_transform(if x.is_conjugated() {
                BlasTransform::Conjugate
            } else {
                BlasTransform::None
            }));
            check_vector(y.len(), y.stride(), y.values().len());
            integer(y.len());
            integer(y.stride());
            assert_eq!(a.rows(), a.cols());
            assert_eq!(a.rows(), x.len());
            assert_eq!(x.len(), y.len());
        }
        BlasCall::Hemv { a, x, y, .. } => {
            matrix(*a);
            check_vector(x.len(), x.stride(), x.values().len());
            integer(x.len());
            integer(x.stride());
            assert!(caps.supports_transform(if x.is_conjugated() {
                BlasTransform::Conjugate
            } else {
                BlasTransform::None
            }));
            check_vector(y.len(), y.stride(), y.values().len());
            integer(y.len());
            integer(y.stride());
            assert_eq!(a.rows(), a.cols());
            assert_eq!(a.rows(), x.len());
            assert_eq!(x.len(), y.len());
        }
        BlasCall::Sbmv { a, x, y, .. } => {
            assert!(caps.supports_layout(BlasLayout::ColumnMajor));
            assert!(a.leading_dimension() > a.lower() + a.upper());
            for n in [
                a.rows(),
                a.cols(),
                a.lower(),
                a.upper(),
                a.leading_dimension(),
            ] {
                integer(n);
            }
            for col in 0..a.cols() {
                for row in col.saturating_sub(a.upper())
                    ..a.rows()
                        .min(col.saturating_add(a.lower()).saturating_add(1))
                {
                    assert!(col * a.leading_dimension() + a.upper() + row - col < a.values().len());
                }
            }
            check_vector(x.len(), x.stride(), x.values().len());
            integer(x.len());
            integer(x.stride());
            assert!(caps.supports_transform(if x.is_conjugated() {
                BlasTransform::Conjugate
            } else {
                BlasTransform::None
            }));
            check_vector(y.len(), y.stride(), y.values().len());
            integer(y.len());
            integer(y.stride());
            assert_eq!(a.rows(), a.cols());
            assert_eq!(a.rows(), x.len());
            assert_eq!(x.len(), y.len());
        }
        BlasCall::Hbmv { a, x, y, .. } => {
            assert!(caps.supports_layout(BlasLayout::ColumnMajor));
            assert!(a.leading_dimension() > a.lower() + a.upper());
            for n in [
                a.rows(),
                a.cols(),
                a.lower(),
                a.upper(),
                a.leading_dimension(),
            ] {
                integer(n);
            }
            for col in 0..a.cols() {
                for row in col.saturating_sub(a.upper())
                    ..a.rows()
                        .min(col.saturating_add(a.lower()).saturating_add(1))
                {
                    assert!(col * a.leading_dimension() + a.upper() + row - col < a.values().len());
                }
            }
            check_vector(x.len(), x.stride(), x.values().len());
            integer(x.len());
            integer(x.stride());
            assert!(caps.supports_transform(if x.is_conjugated() {
                BlasTransform::Conjugate
            } else {
                BlasTransform::None
            }));
            check_vector(y.len(), y.stride(), y.values().len());
            integer(y.len());
            integer(y.stride());
            assert_eq!(a.rows(), a.cols());
            assert_eq!(a.rows(), x.len());
            assert_eq!(x.len(), y.len());
        }
        BlasCall::Spmv { a, x, y, .. } => {
            assert!(caps.supports_layout(BlasLayout::ColumnMajor));
            integer(a.order());
            assert!(a.values().len() >= a.order() * (a.order() + 1) / 2);
            check_vector(x.len(), x.stride(), x.values().len());
            integer(x.len());
            integer(x.stride());
            assert!(caps.supports_transform(if x.is_conjugated() {
                BlasTransform::Conjugate
            } else {
                BlasTransform::None
            }));
            check_vector(y.len(), y.stride(), y.values().len());
            integer(y.len());
            integer(y.stride());
            assert_eq!(a.order(), x.len());
            assert_eq!(x.len(), y.len());
        }
        BlasCall::Hpmv { a, x, y, .. } => {
            assert!(caps.supports_layout(BlasLayout::ColumnMajor));
            integer(a.order());
            assert!(a.values().len() >= a.order() * (a.order() + 1) / 2);
            check_vector(x.len(), x.stride(), x.values().len());
            integer(x.len());
            integer(x.stride());
            assert!(caps.supports_transform(if x.is_conjugated() {
                BlasTransform::Conjugate
            } else {
                BlasTransform::None
            }));
            check_vector(y.len(), y.stride(), y.values().len());
            integer(y.len());
            integer(y.stride());
            assert_eq!(a.order(), x.len());
            assert_eq!(x.len(), y.len());
        }
        BlasCall::Trsv {
            transform, a, x, ..
        } => {
            check_transform(*transform);
            matrix(*a);
            check_vector(x.len(), x.stride(), x.values().len());
            integer(x.len());
            integer(x.stride());
            assert_eq!(a.rows(), a.cols());
            assert_eq!(a.rows(), x.len());
        }
        BlasCall::Tbsv {
            transform, a, x, ..
        } => {
            check_transform(*transform);
            assert!(caps.supports_layout(BlasLayout::ColumnMajor));
            assert!(a.leading_dimension() > a.lower() + a.upper());
            for n in [
                a.rows(),
                a.cols(),
                a.lower(),
                a.upper(),
                a.leading_dimension(),
            ] {
                integer(n);
            }
            for col in 0..a.cols() {
                for row in col.saturating_sub(a.upper())
                    ..a.rows()
                        .min(col.saturating_add(a.lower()).saturating_add(1))
                {
                    assert!(col * a.leading_dimension() + a.upper() + row - col < a.values().len());
                }
            }
            check_vector(x.len(), x.stride(), x.values().len());
            integer(x.len());
            integer(x.stride());
            assert_eq!(a.rows(), a.cols());
            assert_eq!(a.rows(), x.len());
        }
        BlasCall::Tpsv {
            transform, a, x, ..
        } => {
            check_transform(*transform);
            assert!(caps.supports_layout(BlasLayout::ColumnMajor));
            integer(a.order());
            assert!(a.values().len() >= a.order() * (a.order() + 1) / 2);
            check_vector(x.len(), x.stride(), x.values().len());
            integer(x.len());
            integer(x.stride());
            assert_eq!(a.order(), x.len());
        }
        BlasCall::Geru { x, y, a, .. } => {
            check_vector(x.len(), x.stride(), x.values().len());
            integer(x.len());
            integer(x.stride());
            assert!(caps.supports_transform(if x.is_conjugated() {
                BlasTransform::Conjugate
            } else {
                BlasTransform::None
            }));
            check_vector(y.len(), y.stride(), y.values().len());
            integer(y.len());
            integer(y.stride());
            assert!(caps.supports_transform(if y.is_conjugated() {
                BlasTransform::Conjugate
            } else {
                BlasTransform::None
            }));
            output(a);
            assert_eq!((a.rows(), a.cols()), (x.len(), y.len()));
        }
        BlasCall::Gerc { x, y, a, .. } => {
            check_vector(x.len(), x.stride(), x.values().len());
            integer(x.len());
            integer(x.stride());
            assert!(caps.supports_transform(if x.is_conjugated() {
                BlasTransform::Conjugate
            } else {
                BlasTransform::None
            }));
            check_vector(y.len(), y.stride(), y.values().len());
            integer(y.len());
            integer(y.stride());
            assert!(caps.supports_transform(if y.is_conjugated() {
                BlasTransform::Conjugate
            } else {
                BlasTransform::None
            }));
            output(a);
            assert_eq!((a.rows(), a.cols()), (x.len(), y.len()));
        }
        BlasCall::Syr { x, a, .. } => {
            check_vector(x.len(), x.stride(), x.values().len());
            integer(x.len());
            integer(x.stride());
            assert!(caps.supports_transform(if x.is_conjugated() {
                BlasTransform::Conjugate
            } else {
                BlasTransform::None
            }));
            output(a);
            assert_eq!(a.rows(), a.cols());
            assert_eq!(a.rows(), x.len());
        }
        BlasCall::Syr2 { x, y, a, .. } => {
            check_vector(x.len(), x.stride(), x.values().len());
            integer(x.len());
            integer(x.stride());
            assert!(caps.supports_transform(if x.is_conjugated() {
                BlasTransform::Conjugate
            } else {
                BlasTransform::None
            }));
            check_vector(y.len(), y.stride(), y.values().len());
            integer(y.len());
            integer(y.stride());
            assert!(caps.supports_transform(if y.is_conjugated() {
                BlasTransform::Conjugate
            } else {
                BlasTransform::None
            }));
            output(a);
            assert_eq!(a.rows(), a.cols());
            assert_eq!(a.rows(), x.len());
            assert_eq!(x.len(), y.len());
        }
        BlasCall::Her { x, a, .. } => {
            check_vector(x.len(), x.stride(), x.values().len());
            integer(x.len());
            integer(x.stride());
            assert!(caps.supports_transform(if x.is_conjugated() {
                BlasTransform::Conjugate
            } else {
                BlasTransform::None
            }));
            output(a);
            assert_eq!(a.rows(), a.cols());
            assert_eq!(a.rows(), x.len());
        }
        BlasCall::Her2 { x, y, a, .. } => {
            check_vector(x.len(), x.stride(), x.values().len());
            integer(x.len());
            integer(x.stride());
            assert!(caps.supports_transform(if x.is_conjugated() {
                BlasTransform::Conjugate
            } else {
                BlasTransform::None
            }));
            check_vector(y.len(), y.stride(), y.values().len());
            integer(y.len());
            integer(y.stride());
            assert!(caps.supports_transform(if y.is_conjugated() {
                BlasTransform::Conjugate
            } else {
                BlasTransform::None
            }));
            output(a);
            assert_eq!(a.rows(), a.cols());
            assert_eq!(a.rows(), x.len());
            assert_eq!(x.len(), y.len());
        }
        BlasCall::Spr { x, a, .. } => {
            check_vector(x.len(), x.stride(), x.values().len());
            integer(x.len());
            integer(x.stride());
            assert!(caps.supports_transform(if x.is_conjugated() {
                BlasTransform::Conjugate
            } else {
                BlasTransform::None
            }));
            assert!(caps.supports_layout(BlasLayout::ColumnMajor));
            integer(a.order());
            assert!(a.values().len() >= a.order() * (a.order() + 1) / 2);
            assert_eq!(a.order(), x.len());
        }
        BlasCall::Spr2 { x, y, a, .. } => {
            check_vector(x.len(), x.stride(), x.values().len());
            integer(x.len());
            integer(x.stride());
            assert!(caps.supports_transform(if x.is_conjugated() {
                BlasTransform::Conjugate
            } else {
                BlasTransform::None
            }));
            check_vector(y.len(), y.stride(), y.values().len());
            integer(y.len());
            integer(y.stride());
            assert!(caps.supports_transform(if y.is_conjugated() {
                BlasTransform::Conjugate
            } else {
                BlasTransform::None
            }));
            assert!(caps.supports_layout(BlasLayout::ColumnMajor));
            integer(a.order());
            assert!(a.values().len() >= a.order() * (a.order() + 1) / 2);
            assert_eq!(a.order(), x.len());
            assert_eq!(x.len(), y.len());
        }
        BlasCall::Hpr { x, a, .. } => {
            check_vector(x.len(), x.stride(), x.values().len());
            integer(x.len());
            integer(x.stride());
            assert!(caps.supports_transform(if x.is_conjugated() {
                BlasTransform::Conjugate
            } else {
                BlasTransform::None
            }));
            assert!(caps.supports_layout(BlasLayout::ColumnMajor));
            integer(a.order());
            assert!(a.values().len() >= a.order() * (a.order() + 1) / 2);
            assert_eq!(a.order(), x.len());
        }
        BlasCall::Hpr2 { x, y, a, .. } => {
            check_vector(x.len(), x.stride(), x.values().len());
            integer(x.len());
            integer(x.stride());
            assert!(caps.supports_transform(if x.is_conjugated() {
                BlasTransform::Conjugate
            } else {
                BlasTransform::None
            }));
            check_vector(y.len(), y.stride(), y.values().len());
            integer(y.len());
            integer(y.stride());
            assert!(caps.supports_transform(if y.is_conjugated() {
                BlasTransform::Conjugate
            } else {
                BlasTransform::None
            }));
            assert!(caps.supports_layout(BlasLayout::ColumnMajor));
            integer(a.order());
            assert!(a.values().len() >= a.order() * (a.order() + 1) / 2);
            assert_eq!(a.order(), x.len());
            assert_eq!(x.len(), y.len());
        }
        BlasCall::Symm { a, b, c, side, .. } => {
            matrix(*a);
            matrix(*b);
            output(c);
            assert_eq!(a.layout(), c.layout());
            assert_eq!(b.layout(), c.layout());
            assert_eq!(a.rows(), a.cols());
            assert_eq!(
                a.rows(),
                if *side == MatrixSide::Left {
                    b.rows()
                } else {
                    b.cols()
                }
            );
            assert_eq!((b.rows(), b.cols()), (c.rows(), c.cols()));
        }
        BlasCall::Hemm { a, b, c, side, .. } => {
            matrix(*a);
            matrix(*b);
            output(c);
            assert_eq!(a.layout(), c.layout());
            assert_eq!(b.layout(), c.layout());
            assert_eq!(a.rows(), a.cols());
            assert_eq!(
                a.rows(),
                if *side == MatrixSide::Left {
                    b.rows()
                } else {
                    b.cols()
                }
            );
            assert_eq!((b.rows(), b.cols()), (c.rows(), c.cols()));
        }
        BlasCall::Syr2k {
            transform, a, b, c, ..
        } => {
            check_transform(*transform);
            matrix(*a);
            matrix(*b);
            output(c);
            assert_eq!(a.layout(), c.layout());
            assert_eq!(b.layout(), c.layout());
            assert_eq!(c.rows(), c.cols());
            assert_eq!(matrix_logical_shape(*a, *transform).0, c.rows());
            assert!(matches!(
                transform,
                BlasTransform::None | BlasTransform::Transpose
            ));
            assert_eq!((a.rows(), a.cols()), (b.rows(), b.cols()));
        }
        BlasCall::Her2k {
            transform, a, b, c, ..
        } => {
            check_transform(*transform);
            matrix(*a);
            matrix(*b);
            output(c);
            assert_eq!(a.layout(), c.layout());
            assert_eq!(b.layout(), c.layout());
            assert_eq!(c.rows(), c.cols());
            assert_eq!(matrix_logical_shape(*a, *transform).0, c.rows());
            assert!(matches!(
                transform,
                BlasTransform::None | BlasTransform::Adjoint
            ));
            assert_eq!((a.rows(), a.cols()), (b.rows(), b.cols()));
        }
    }
}

fn check_dense(rows: usize, cols: usize, ld: usize, layout: BlasLayout, len: usize) {
    let (major, minor) = match layout {
        BlasLayout::RowMajor => (rows, cols),
        BlasLayout::ColumnMajor => (cols, rows),
    };
    assert!(
        ld >= minor.max(1),
        "invalid CBLAS leading dimension {ld} for {rows}x{cols}"
    );
    if major != 0 && minor != 0 {
        assert!((major - 1) * ld + minor <= len);
    }
}
fn check_vector(len: usize, stride: usize, span: usize) {
    assert!(stride > 0);
    if len != 0 {
        assert!((len - 1) * stride < span);
    }
}

fn scalar<S: NumericScalar>(real: f64, imaginary: f64) -> S {
    S::from_components(
        if real.is_nan() {
            S::Real::NAN
        } else {
            S::Real::try_from_f64_lossy(real).unwrap()
        },
        S::Real::try_from_f64_lossy(if S::IS_REAL { 0.0 } else { imaginary }).unwrap(),
    )
}
fn close<S: NumericScalar>(actual: &[S], expected: &[S], work: usize) {
    assert_eq!(actual.len(), expected.len());
    let bound = 128.0 * work.max(1) as f64 * S::Real::EPSILON.into_f64();
    for (&a, &e) in actual.iter().zip(expected) {
        if !e.has_finite_components() {
            assert!(!a.has_finite_components());
        } else {
            assert!(
                (a - e).abs().into_f64() <= bound * (1.0 + e.abs().into_f64()),
                "{a:?} != {e:?}"
            );
        }
    }
}
fn assert_routed(provider: &CblasProvider, operation: BlasOperation) {
    assert_ne!(
        provider.operations.load(Ordering::Relaxed) & (1 << operation as u8),
        0,
        "missing {operation:?}"
    );
}

fn layout_provider(layout: BlasLayout) -> CblasProvider {
    let all = CblasProvider::new();
    let caps = BlasCapabilities::empty(BlasIntegerWidth::Lp64)
        .with_layout(layout)
        .with_transform(BlasTransform::None)
        .with_transform(BlasTransform::Transpose)
        .with_transform(BlasTransform::Adjoint)
        .with_transform(BlasTransform::Conjugate);
    // Use the declaration's operation list, without relying on enum transmutation.
    let caps = ALL_OPERATIONS.iter().fold(caps, |caps, &op| {
        [
            ScalarKind::F32,
            ScalarKind::F64,
            ScalarKind::Complex32,
            ScalarKind::Complex64,
        ]
        .into_iter()
        .fold(caps, |caps, scalar| {
            if let Some(ulp) = all.capabilities.operation_ulp(scalar, op) {
                caps.with_operation(scalar, op, ulp).unwrap()
            } else {
                caps
            }
        })
    });
    CblasProvider::with_capabilities(caps)
}

pub(super) const ALL_OPERATIONS: [BlasOperation; 45] = [
    BlasOperation::Copy,
    BlasOperation::Scale,
    BlasOperation::Axpy,
    BlasOperation::Dotu,
    BlasOperation::Dotc,
    BlasOperation::Norm2,
    BlasOperation::Gemv,
    BlasOperation::Gemm,
    BlasOperation::Syrk,
    BlasOperation::Herk,
    BlasOperation::Trmm,
    BlasOperation::Trsm,
    BlasOperation::Swap,
    BlasOperation::Asum,
    BlasOperation::Iamax,
    BlasOperation::Rotg,
    BlasOperation::Rotmg,
    BlasOperation::Rot,
    BlasOperation::Rotm,
    BlasOperation::WideDot,
    BlasOperation::OffsetDot,
    BlasOperation::Gbmv,
    BlasOperation::Symv,
    BlasOperation::Hemv,
    BlasOperation::Sbmv,
    BlasOperation::Hbmv,
    BlasOperation::Spmv,
    BlasOperation::Hpmv,
    BlasOperation::Trsv,
    BlasOperation::Tbsv,
    BlasOperation::Tpsv,
    BlasOperation::Geru,
    BlasOperation::Gerc,
    BlasOperation::Syr,
    BlasOperation::Syr2,
    BlasOperation::Her,
    BlasOperation::Her2,
    BlasOperation::Spr,
    BlasOperation::Spr2,
    BlasOperation::Hpr,
    BlasOperation::Hpr2,
    BlasOperation::Symm,
    BlasOperation::Hemm,
    BlasOperation::Syr2k,
    BlasOperation::Her2k,
];

fn stored_index(layout: BlasLayout, n: usize, row: usize, col: usize) -> usize {
    match layout {
        BlasLayout::RowMajor => row * (n + 2) + col,
        BlasLayout::ColumnMajor => col * (n + 2) + row,
    }
}
fn stored_view<S: NumericScalar>(data: &[S], n: usize, layout: BlasLayout) -> MatrixView<'_, S> {
    let (rs, cs) = match layout {
        BlasLayout::RowMajor => (n + 2, 1),
        BlasLayout::ColumnMajor => (1, n + 2),
    };
    MatrixView::from_slice_with_strides(n, n, data, rs, cs).unwrap()
}

fn triangular_combinations<S: NumericScalar>() {
    let options = LinalgExecutionOptions::new();
    for provider_layout in [BlasLayout::RowMajor, BlasLayout::ColumnMajor] {
        for storage in [BlasLayout::RowMajor, BlasLayout::ColumnMajor] {
            for triangle in [Triangle::Lower, Triangle::Upper] {
                for diagonal in [DiagonalKind::Explicit, DiagonalKind::Unit] {
                    for op in [
                        StructuredOperation::Normal,
                        StructuredOperation::Transpose,
                        StructuredOperation::Adjoint,
                    ] {
                        let n = 4;
                        let mut data = vec![scalar::<S>(f64::NAN, 0.0); n * (n + 2)];
                        for row in 0..n {
                            for col in 0..n {
                                let chosen = match triangle {
                                    Triangle::Lower => row >= col,
                                    Triangle::Upper => row <= col,
                                };
                                if chosen && (row != col || diagonal != DiagonalKind::Unit) {
                                    data[stored_index(storage, n, row, col)] = scalar(
                                        if row == col {
                                            2.0 + row as f64
                                        } else {
                                            (row + col + 1) as f64 / 8.0
                                        },
                                        (row as f64 - col as f64) / 7.0,
                                    );
                                }
                            }
                        }
                        let a = stored_view(&data, n, storage);
                        for side in [MatrixSide::Left, MatrixSide::Right] {
                            for solve in [false, true] {
                                let provider = layout_provider(provider_layout);
                                let (m, k) = if side == MatrixSide::Left {
                                    (n, 3)
                                } else {
                                    (3, n)
                                };
                                let initial = (0..m * k)
                                    .map(|i| {
                                        scalar::<S>((i + 1) as f64 / 9.0, (i % 3) as f64 / 5.0)
                                    })
                                    .collect::<Vec<_>>();
                                let mut actual = initial.clone();
                                let mut expected = initial;
                                for (out, context) in [
                                    (&mut expected, LinalgExecutionContext::sequential()),
                                    (&mut actual, provider.context()),
                                ] {
                                    let mut workspace = LinalgWorkspace::new();
                                    let output = MatrixViewMut::from_slice(m, k, out).unwrap();
                                    if solve {
                                        a.triangular_solve_mat_in_place(
                                            side,
                                            triangle,
                                            diagonal,
                                            op,
                                            scalar(0.7, 0.2),
                                            &options,
                                            output,
                                            &mut workspace,
                                            context,
                                        )
                                        .unwrap();
                                    } else {
                                        a.triangular_mul_mat_in_place(
                                            side,
                                            triangle,
                                            diagonal,
                                            op,
                                            scalar(0.7, 0.2),
                                            &options,
                                            output,
                                            &mut workspace,
                                            context,
                                        )
                                        .unwrap();
                                    }
                                }
                                close(&actual, &expected, n);
                                assert_routed(
                                    &provider,
                                    if solve {
                                        BlasOperation::Trsm
                                    } else {
                                        BlasOperation::Trmm
                                    },
                                );
                            }
                        }
                        // Level 2 keeps the input's storage layout, with the same stored Uplo semantics.
                        let provider = layout_provider(storage);
                        let plan = TriangularPlan::from_dense(a, triangle, diagonal).unwrap();
                        let input = (0..n)
                            .map(|i| scalar::<S>(i as f64 + 1.0, 0.3))
                            .collect::<Vec<_>>();
                        let mut actual = vec![S::ZERO; n];
                        let mut expected = actual.clone();
                        for (out, context) in [
                            (&mut expected, LinalgExecutionContext::sequential()),
                            (&mut actual, provider.context()),
                        ] {
                            let _ = plan
                                .solve_into(
                                    op,
                                    VectorView::from_slice(&input),
                                    &DirectSolveOptions::new(RankPolicy::Exact),
                                    VectorViewMut::from_slice(out),
                                    &mut LinalgWorkspace::new(),
                                    context,
                                )
                                .unwrap();
                        }
                        close(&actual, &expected, n);
                        assert_routed(&provider, BlasOperation::Trsv);
                    }
                }
            }
        }
    }
}

#[test]
fn triangular_cblas_cartesian_product() {
    triangular_combinations::<f32>();
    triangular_combinations::<f64>();
    triangular_combinations::<Complex32>();
    triangular_combinations::<Complex64>();
}

fn self_adjoint_combinations<S: NumericScalar>() {
    for layout in [BlasLayout::RowMajor, BlasLayout::ColumnMajor] {
        for storage in [BlasLayout::RowMajor, BlasLayout::ColumnMajor] {
            for triangle in [Triangle::Lower, Triangle::Upper] {
                for op in [
                    StructuredOperation::Normal,
                    StructuredOperation::Transpose,
                    StructuredOperation::Adjoint,
                ] {
                    for side in [MatrixSide::Left, MatrixSide::Right] {
                        for hermitian in [false, true] {
                            let n = 4;
                            let provider = layout_provider(layout);
                            let mut data = vec![scalar::<S>(f64::NAN, 0.0); n * (n + 2)];
                            for row in 0..n {
                                for col in 0..n {
                                    let selected = match triangle {
                                        Triangle::Lower => row >= col,
                                        Triangle::Upper => row <= col,
                                    };
                                    if selected {
                                        let (r, c) = if op == StructuredOperation::Normal {
                                            (row, col)
                                        } else {
                                            (col, row)
                                        };
                                        data[stored_index(storage, n, r, c)] = scalar(
                                            (row + col + 1) as f64 / 3.0,
                                            if row == col {
                                                0.0
                                            } else {
                                                (row + 1) as f64 / 7.0
                                            },
                                        );
                                    }
                                }
                            }
                            let a = stored_view(&data, n, storage);
                            let a = match op {
                                StructuredOperation::Normal => a,
                                StructuredOperation::Transpose => a.transpose(),
                                StructuredOperation::Adjoint => a.adjoint(),
                            };
                            let (rows, cols) = if side == MatrixSide::Left {
                                (n, 3)
                            } else {
                                (3, n)
                            };
                            let b = (0..rows * cols)
                                .map(|i| scalar::<S>((i + 1) as f64 / 8.0, (i % 3) as f64 / 5.0))
                                .collect::<Vec<_>>();
                            let b = MatrixView::from_slice(rows, cols, &b).unwrap();
                            let mut actual = vec![scalar::<S>(0.3, 0.2); rows * cols];
                            let mut expected = actual.clone();
                            for (out, context) in [
                                (&mut expected, LinalgExecutionContext::sequential()),
                                (&mut actual, provider.context()),
                            ] {
                                let mut workspace = LinalgWorkspace::new();
                                let output = MatrixViewMut::from_slice(rows, cols, out).unwrap();
                                if hermitian {
                                    a.hermitian_mul_mat_in_place(
                                        b,
                                        side,
                                        triangle,
                                        scalar(0.7, 0.2),
                                        scalar(0.3, 0.1),
                                        &LinalgExecutionOptions::new(),
                                        output,
                                        &mut workspace,
                                        context,
                                    )
                                    .unwrap();
                                } else {
                                    a.symmetric_mul_mat_in_place(
                                        b,
                                        side,
                                        triangle,
                                        scalar(0.7, 0.2),
                                        scalar(0.3, 0.1),
                                        &LinalgExecutionOptions::new(),
                                        output,
                                        &mut workspace,
                                        context,
                                    )
                                    .unwrap();
                                }
                            }
                            close(&actual, &expected, n);
                            assert_routed(
                                &provider,
                                if hermitian && !S::IS_REAL {
                                    BlasOperation::Hemm
                                } else {
                                    BlasOperation::Symm
                                },
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn self_adjoint_cblas_cartesian_product() {
    self_adjoint_combinations::<f32>();
    self_adjoint_combinations::<f64>();
    self_adjoint_combinations::<Complex32>();
    self_adjoint_combinations::<Complex64>();
}

fn matrix_case<S: NumericScalar>(
    operation: BlasOperation,
    context: LinalgExecutionContext<'_, '_>,
) -> Vec<S> {
    use BlasOperation::*;
    use scioxide_linalg::{
        HermitianBandMatrixView, PackedKind, PackedMatrixView, PackedMatrixViewMut,
    };
    let options = LinalgExecutionOptions::new();
    let mut workspace = LinalgWorkspace::new();
    let data = [
        scalar(2.0, 0.0),
        S::ZERO,
        S::ZERO,
        scalar(0.5, 0.2),
        scalar(3.0, 0.0),
        S::ZERO,
        scalar(0.25, -0.1),
        scalar(0.75, 0.3),
        scalar(4.0, 0.0),
    ];
    let band_data = [
        data[0],
        data[3],
        data[6],
        data[4],
        data[7],
        S::ZERO,
        data[8],
        S::ZERO,
        S::ZERO,
    ];
    let packed_data = [data[0], data[3], data[6], data[4], data[7], data[8]];
    let x_data = [scalar(0.3, 0.7), scalar(-0.5, 0.2), scalar(0.9, -0.1)];
    let y_data = [scalar(0.5, -0.3), scalar(0.4, 0.8), scalar(-0.7, 0.2)];
    let a = MatrixView::from_slice(3, 3, &data).unwrap();
    let x = VectorView::from_slice(&x_data);
    let y = VectorView::from_slice(&y_data);
    let alpha = scalar(0.7, 0.0);
    let beta = scalar(0.3, 0.0);
    let mut out = vec![scalar(0.4, 0.0); 9];
    match operation {
        Gemv => {
            out.truncate(3);
            a.mul_vec_into(
                x,
                &options,
                VectorViewMut::from_slice(&mut out),
                &mut workspace,
                context,
            )
            .unwrap();
        }
        Gbmv => {
            out.truncate(3);
            BandedMatrixView::new(3, 3, 2, 0, 3, &band_data)
                .unwrap()
                .mul_vec_in_place(
                    StructuredOperation::Normal,
                    alpha,
                    x,
                    beta,
                    &options,
                    VectorViewMut::from_slice(&mut out),
                    &mut workspace,
                    context,
                )
                .unwrap();
        }
        Symv | Hemv => {
            out.truncate(3);
            if operation == Symv {
                a.symmetric_mul_vec_in_place(
                    Triangle::Lower,
                    alpha,
                    x,
                    beta,
                    &options,
                    VectorViewMut::from_slice(&mut out),
                    &mut workspace,
                    context,
                )
                .unwrap();
            } else {
                a.hermitian_mul_vec_in_place(
                    Triangle::Lower,
                    alpha,
                    x,
                    beta,
                    &options,
                    VectorViewMut::from_slice(&mut out),
                    &mut workspace,
                    context,
                )
                .unwrap();
            }
        }
        Sbmv | Hbmv => {
            out.truncate(3);
            HermitianBandMatrixView::new(3, 2, Triangle::Lower, &band_data)
                .unwrap()
                .mul_vec_in_place(
                    alpha,
                    x,
                    beta,
                    &options,
                    VectorViewMut::from_slice(&mut out),
                    &mut workspace,
                    context,
                )
                .unwrap();
        }
        Spmv | Hpmv => {
            out.truncate(3);
            PackedMatrixView::new(3, Triangle::Lower, PackedKind::Hermitian, &packed_data)
                .unwrap()
                .mul_vec_in_place(
                    StructuredOperation::Normal,
                    alpha,
                    x,
                    beta,
                    &options,
                    VectorViewMut::from_slice(&mut out),
                    &mut workspace,
                    context,
                )
                .unwrap();
        }
        Trsv | Tbsv | Tpsv => {
            out.truncate(3);
            let plan = match operation {
                Trsv => TriangularPlan::from_dense(a, Triangle::Lower, DiagonalKind::Explicit),
                Tbsv => TriangularPlan::from_band(
                    BandedMatrixView::new(3, 3, 2, 0, 3, &band_data).unwrap(),
                    Triangle::Lower,
                    DiagonalKind::Explicit,
                ),
                _ => TriangularPlan::from_packed(
                    PackedMatrixView::new(3, Triangle::Lower, PackedKind::Triangular, &packed_data)
                        .unwrap(),
                    DiagonalKind::Explicit,
                ),
            }
            .unwrap();
            let _ = plan
                .solve_into(
                    StructuredOperation::Adjoint,
                    x,
                    &DirectSolveOptions::new(RankPolicy::Exact),
                    VectorViewMut::from_slice(&mut out),
                    &mut workspace,
                    context,
                )
                .unwrap();
        }
        Spr | Spr2 | Hpr | Hpr2 => {
            out = packed_data.to_vec();
            let output =
                PackedMatrixViewMut::new(3, Triangle::Lower, PackedKind::Hermitian, &mut out)
                    .unwrap();
            if matches!(operation, Spr | Hpr) {
                PackedMatrixViewMut::hermitian_rank_one_update_in_place(
                    alpha,
                    x,
                    &options,
                    output,
                    &mut workspace,
                    context,
                )
                .unwrap();
            } else {
                PackedMatrixViewMut::hermitian_rank_two_update_in_place(
                    alpha,
                    x,
                    y,
                    &options,
                    output,
                    &mut workspace,
                    context,
                )
                .unwrap();
            }
        }
        Geru => MatrixViewMut::rank_one_update_in_place(
            alpha,
            x,
            y,
            &options,
            MatrixViewMut::from_slice(3, 3, &mut out).unwrap(),
            &mut workspace,
            context,
        )
        .unwrap(),
        Gerc => MatrixViewMut::rank_one_adjoint_update_in_place(
            alpha,
            x,
            y,
            &options,
            MatrixViewMut::from_slice(3, 3, &mut out).unwrap(),
            &mut workspace,
            context,
        )
        .unwrap(),
        Syr => MatrixViewMut::symmetric_rank_one_update_in_place(
            Triangle::Lower,
            alpha,
            x,
            &options,
            MatrixViewMut::from_slice(3, 3, &mut out).unwrap(),
            &mut workspace,
            context,
        )
        .unwrap(),
        Her => MatrixViewMut::hermitian_rank_one_update_in_place(
            Triangle::Lower,
            alpha,
            x,
            &options,
            MatrixViewMut::from_slice(3, 3, &mut out).unwrap(),
            &mut workspace,
            context,
        )
        .unwrap(),
        Syr2 => MatrixViewMut::symmetric_rank_two_update_in_place(
            Triangle::Lower,
            alpha,
            x,
            y,
            &options,
            MatrixViewMut::from_slice(3, 3, &mut out).unwrap(),
            &mut workspace,
            context,
        )
        .unwrap(),
        Her2 => MatrixViewMut::hermitian_rank_two_update_in_place(
            Triangle::Lower,
            alpha,
            x,
            y,
            &options,
            MatrixViewMut::from_slice(3, 3, &mut out).unwrap(),
            &mut workspace,
            context,
        )
        .unwrap(),
        Gemm => a
            .mul_mat_into(
                a.transpose(),
                &options,
                MatrixViewMut::from_slice(3, 3, &mut out).unwrap(),
                &mut workspace,
                context,
            )
            .unwrap(),
        Syrk => scioxide_linalg::MatrixViewMut::symmetric_rank_k_in_place(
            a,
            Triangle::Lower,
            StructuredOperation::Transpose,
            alpha,
            beta,
            &options,
            MatrixViewMut::from_slice(3, 3, &mut out).unwrap(),
            &mut workspace,
            context,
        )
        .unwrap(),
        Herk => scioxide_linalg::MatrixViewMut::hermitian_rank_k_in_place(
            a,
            Triangle::Lower,
            StructuredOperation::Adjoint,
            alpha,
            beta,
            &options,
            MatrixViewMut::from_slice(3, 3, &mut out).unwrap(),
            &mut workspace,
            context,
        )
        .unwrap(),
        Syr2k => scioxide_linalg::MatrixViewMut::symmetric_rank_two_k_in_place(
            a,
            a,
            Triangle::Lower,
            StructuredOperation::Transpose,
            alpha,
            beta,
            &options,
            MatrixViewMut::from_slice(3, 3, &mut out).unwrap(),
            &mut workspace,
            context,
        )
        .unwrap(),
        Her2k => scioxide_linalg::MatrixViewMut::hermitian_rank_two_k_in_place(
            a,
            a,
            Triangle::Lower,
            StructuredOperation::Adjoint,
            alpha,
            beta,
            &options,
            MatrixViewMut::from_slice(3, 3, &mut out).unwrap(),
            &mut workspace,
            context,
        )
        .unwrap(),
        Symm => a
            .symmetric_mul_mat_in_place(
                a,
                MatrixSide::Left,
                Triangle::Lower,
                alpha,
                beta,
                &options,
                MatrixViewMut::from_slice(3, 3, &mut out).unwrap(),
                &mut workspace,
                context,
            )
            .unwrap(),
        Hemm => a
            .hermitian_mul_mat_in_place(
                a,
                MatrixSide::Left,
                Triangle::Lower,
                alpha,
                beta,
                &options,
                MatrixViewMut::from_slice(3, 3, &mut out).unwrap(),
                &mut workspace,
                context,
            )
            .unwrap(),
        Trmm => a
            .triangular_mul_mat_in_place(
                MatrixSide::Left,
                Triangle::Lower,
                DiagonalKind::Explicit,
                StructuredOperation::Adjoint,
                alpha,
                &options,
                MatrixViewMut::from_slice(3, 3, &mut out).unwrap(),
                &mut workspace,
                context,
            )
            .unwrap(),
        Trsm => a
            .triangular_solve_mat_in_place(
                MatrixSide::Left,
                Triangle::Lower,
                DiagonalKind::Explicit,
                StructuredOperation::Adjoint,
                alpha,
                &options,
                MatrixViewMut::from_slice(3, 3, &mut out).unwrap(),
                &mut workspace,
                context,
            )
            .unwrap(),
        _ => unreachable!(),
    }
    out
}

fn vector_case(operation: BlasOperation, mut context: LinalgExecutionContext<'_, '_>) -> Vec<f64> {
    use BlasOperation::*;
    let options = LinalgExecutionOptions::new();
    let mut workspace = LinalgWorkspace::new();
    let x = VectorView::from_slice(&[1.0_f64, -2.0, 0.5]);
    let y = VectorView::from_slice(&[0.3_f64, 0.7, -0.8]);
    let mut out = vec![0.5; 3];
    match operation {
        Copy => x
            .copy_into(
                scioxide_linalg::CopyPolicy::RequireFinite,
                &options,
                VectorViewMut::from_slice(&mut out),
                &mut workspace,
                context,
            )
            .unwrap(),
        Scale => VectorViewMut::scale_in_place(
            0.7,
            &options,
            VectorViewMut::from_slice(&mut out),
            &mut workspace,
            context,
        )
        .unwrap(),
        Axpy => x
            .axpy_into(
                0.7,
                y,
                &options,
                VectorViewMut::from_slice(&mut out),
                &mut workspace,
                context,
            )
            .unwrap(),
        Dotu => {
            return vec![
                x.dotu_scoped(y, &scioxide_linalg::LinalgExecutionOptions::new(), context)
                    .unwrap(),
            ];
        }
        Dotc => {
            return vec![
                x.dotc_left_scoped(y, &scioxide_linalg::LinalgExecutionOptions::new(), context)
                    .unwrap(),
            ];
        }
        Norm2 => {
            return vec![
                x.norm2_scoped(&scioxide_linalg::LinalgExecutionOptions::new(), context)
                    .unwrap(),
            ];
        }
        Asum => {
            return vec![
                x.asum_scoped(&scioxide_linalg::LinalgExecutionOptions::new(), context)
                    .unwrap(),
            ];
        }
        Iamax => {
            return vec![
                x.iamax_scoped(&scioxide_linalg::LinalgExecutionOptions::new(), context)
                    .unwrap()
                    .unwrap() as f64,
            ];
        }
        WideDot => {
            return vec![
                VectorView::from_slice(&[1.0_f32, -2.0, 0.5])
                    .dot_wide_scoped(
                        VectorView::from_slice(&[0.3_f32, 0.7, -0.8]),
                        &scioxide_linalg::LinalgExecutionOptions::new(),
                        context,
                    )
                    .unwrap(),
            ];
        }
        OffsetDot => {
            return vec![
                VectorView::from_slice(&[1.0_f32, -2.0, 0.5])
                    .dot_plus_scoped(
                        VectorView::from_slice(&[0.3_f32, 0.7, -0.8]),
                        0.25,
                        &scioxide_linalg::LinalgExecutionOptions::new(),
                        context,
                    )
                    .unwrap() as f64,
            ];
        }
        Rotg => {
            let r = scioxide_linalg::givens_rotation_scoped(
                3.0,
                4.0,
                &scioxide_linalg::LinalgExecutionOptions::new(),
                context,
            )
            .unwrap();
            return vec![r.r(), r.z(), r.rotation().cosine(), r.rotation().sine()];
        }
        Rotmg => {
            let r = scioxide_linalg::modified_givens_rotation_scoped(
                1.0,
                1.0,
                3.0,
                4.0,
                &scioxide_linalg::LinalgExecutionOptions::new(),
                context,
            )
            .unwrap();
            return vec![r.d1(), r.d2(), r.x1()];
        }
        Swap | Rot | Rotm => {
            let mut left = vec![1.0, -2.0, 0.5];
            let mut right = vec![0.3, 0.7, -0.8];
            if operation == Swap {
                VectorViewMut::swap_in_place(
                    &options,
                    VectorViewMut::from_slice(&mut left),
                    VectorViewMut::from_slice(&mut right),
                    &mut workspace,
                    context,
                )
                .unwrap();
            } else if operation == Rot {
                VectorViewMut::rotate_in_place(
                    scioxide_linalg::GivensRotation::new(0.6, 0.8).unwrap(),
                    &options,
                    VectorViewMut::from_slice(&mut left),
                    VectorViewMut::from_slice(&mut right),
                    &mut workspace,
                    context,
                )
                .unwrap();
            } else {
                let r = scioxide_linalg::modified_givens_rotation_scoped(
                    1.0,
                    1.0,
                    3.0,
                    4.0,
                    &scioxide_linalg::LinalgExecutionOptions::new(),
                    context.reborrow(),
                )
                .unwrap();
                VectorViewMut::modified_rotate_in_place(
                    r.rotation(),
                    &options,
                    VectorViewMut::from_slice(&mut left),
                    VectorViewMut::from_slice(&mut right),
                    &mut workspace,
                    context,
                )
                .unwrap();
            }
            left.extend(right);
            return left;
        }
        _ => unreachable!(),
    }
    out
}

#[test]
fn all_45_operations_route_and_match_reference() {
    use BlasOperation::*;
    let provider = CblasProvider::new();
    let mut covered = 0;
    for &op in &ALL_OPERATIONS {
        provider.operations.store(0, Ordering::Relaxed);
        match op {
            Copy | Scale | Axpy | Dotu | Dotc | Norm2 | Swap | Asum | Iamax | Rotg | Rotmg
            | Rot | Rotm | WideDot | OffsetDot => {
                let actual = vector_case(op, provider.context());
                let expected = vector_case(op, LinalgExecutionContext::sequential());
                // SDSDOT rounds its wide accumulator back to binary32.
                if op == OffsetDot {
                    assert!((actual[0] - expected[0]).abs() < 1e-6);
                } else {
                    close(&actual, &expected, 3);
                }
            }
            Herk | Hemv | Hbmv | Hpmv | Gerc | Her | Her2 | Hpr | Hpr2 | Hemm | Her2k => {
                check_matrix_case::<Complex32>(&provider, op);
                check_matrix_case::<Complex64>(&provider, op);
            }
            _ => {
                check_matrix_case::<f32>(&provider, op);
                check_matrix_case::<f64>(&provider, op);
                if !matches!(op, Symv | Sbmv | Spmv | Syr | Syr2 | Spr | Spr2) {
                    check_matrix_case::<Complex32>(&provider, op);
                    check_matrix_case::<Complex64>(&provider, op);
                }
            }
        }
        assert_routed(&provider, op);
        covered |= provider.operations.load(Ordering::Relaxed);
    }
    assert_eq!(covered, (1_u64 << 45) - 1);
}

fn random_matrix<S: NumericScalar>(n: usize, hermitian: bool, indefinite: bool) -> Matrix<S> {
    let mut seed = 0x1198_2026_u64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1_u64 << 53) as f64 - 0.5
    };
    let mut data = vec![S::ZERO; n * n];
    for row in 0..n {
        for col in 0..n {
            data[row * n + col] = scalar(next(), next());
        }
    }
    if hermitian {
        for row in 0..n {
            for col in 0..row {
                data[col * n + row] = data[row * n + col].conj();
            }
        }
    }
    for i in 0..n {
        data[i * n + i] = scalar(
            (n + i + 1) as f64 * if indefinite && i % 2 == 0 { -1.0 } else { 1.0 },
            0.0,
        );
    }
    Matrix::from_vec(n, n, data).unwrap()
}

fn blocked_factor_regressions<S: NumericScalar>() {
    use scioxide_linalg::{
        HermitianEigenDecomposition, HermitianEigenOptions, HermitianInput, LdlDecomposition,
    };
    let n = 80;
    let a = random_matrix::<S>(n, true, true);
    let expected = (0..n)
        .map(|i| scalar::<S>((i % 9) as f64 / 7.0, (i % 7) as f64 / 5.0))
        .collect::<Vec<_>>();
    let b = Vector::from_vec(
        (0..n)
            .map(|row| {
                (0..n).fold(S::ZERO, |sum, col| {
                    sum + a.as_slice()[row * n + col] * expected[col]
                })
            })
            .collect(),
    );
    let reference = LdlDecomposition::factor(&a, HermitianInput::Exact).unwrap();
    let native = reference.solve(&b).unwrap();
    for layout in [BlasLayout::RowMajor, BlasLayout::ColumnMajor] {
        let provider = layout_provider(layout);
        let mut factor = LdlDecomposition::empty();
        LdlDecomposition::factor_into(
            &a,
            HermitianInput::Exact,
            &LinalgExecutionOptions::new(),
            &mut factor,
            &mut LinalgWorkspace::new(),
            provider.context(),
        )
        .unwrap();
        let solved = factor.solve(&b).unwrap();
        close(solved.as_slice(), native.as_slice(), n);
        close(solved.as_slice(), &expected, n);
        let l = factor.l_matrix().view();
        for row in 0..n {
            for col in 0..n {
                let mut value = S::ZERO;
                for block in factor.pivot_blocks() {
                    let k = block.start();
                    value = value
                        + l.get(row, k).unwrap()
                            * factor.diagonal()[k]
                            * l.get(col, k).unwrap().conj();
                    if block.size() == 2 {
                        value = value
                            + l.get(row, k + 1).unwrap()
                                * factor.diagonal()[k + 1]
                                * l.get(col, k + 1).unwrap().conj()
                            + l.get(row, k + 1).unwrap()
                                * factor.off_diagonal()[k]
                                * l.get(col, k).unwrap().conj()
                            + l.get(row, k).unwrap()
                                * factor.off_diagonal()[k].conj()
                                * l.get(col, k + 1).unwrap().conj();
                    }
                }
                close(
                    &[value],
                    &[a.view()
                        .get(factor.permutation()[row], factor.permutation()[col])
                        .unwrap()],
                    n,
                );
            }
        }
        assert_routed(&provider, BlasOperation::Gemm);
        assert_eq!(
            provider.operations.load(Ordering::Relaxed)
                & ((1 << BlasOperation::Syrk as u8) | (1 << BlasOperation::Herk as u8)),
            0
        );
    }
    let n = 96;
    let a = random_matrix::<S>(n, true, false);
    let reference = HermitianEigenDecomposition::decompose(&a, HermitianInput::Exact).unwrap();
    for layout in [BlasLayout::RowMajor, BlasLayout::ColumnMajor] {
        let provider = layout_provider(layout);
        let mut result = HermitianEigenDecomposition::empty();
        // A column-only CBLAS cannot express the conjugate-only panel GEMV;
        // permit that documented fallback, while requiring the trailing GEMM below.
        let context = LinalgExecutionContext::sequential().with_blas(BlasDispatch::new(
            &provider,
            if layout == BlasLayout::ColumnMajor {
                BlasFallbackPolicy::Reference
            } else {
                BlasFallbackPolicy::Error
            },
        ));
        HermitianEigenDecomposition::decompose_into(
            &a,
            HermitianInput::Exact,
            &HermitianEigenOptions::default(),
            &mut result,
            &mut LinalgWorkspace::new(),
            context,
        )
        .unwrap();
        close(result.eigenvalues(), reference.eigenvalues(), n);
        let trace = (0..n).fold(S::Real::ZERO, |sum, i| {
            sum + a.as_slice()[i * n + i].real_component()
        });
        let sum = result
            .eigenvalues()
            .iter()
            .copied()
            .fold(S::Real::ZERO, |sum, x| sum + x);
        close(&[sum], &[trace], n);
        let v = result.eigenvectors().view();
        for row in 0..n {
            for col in 0..n {
                let av = (0..n).fold(S::ZERO, |sum, k| {
                    sum + a.as_slice()[row * n + k] * v.get(k, col).unwrap()
                });
                close(
                    &[av],
                    &[v.get(row, col).unwrap() * S::from_real(result.eigenvalues()[col])],
                    n,
                );
                let gram = (0..n).fold(S::ZERO, |sum, k| {
                    sum + v.get(k, row).unwrap().conj() * v.get(k, col).unwrap()
                });
                close(&[gram], &[if row == col { S::ONE } else { S::ZERO }], n);
            }
        }
        assert_routed(&provider, BlasOperation::Gemm);
    }
}

#[test]
fn non_identity_blocked_ldl_and_hermitian_eigen() {
    blocked_factor_regressions::<f64>();
    blocked_factor_regressions::<Complex64>();
}

fn factor_solve_regressions<S: NumericScalar>() {
    let n = 40;
    let spd = random_matrix::<S>(n, true, false);
    let general = random_matrix::<S>(n, false, false);
    let b = Matrix::from_vec(
        n,
        3,
        (0..n * 3)
            .map(|i| scalar::<S>((i % 7) as f64 / 5.0, (i % 5) as f64 / 9.0))
            .collect(),
    )
    .unwrap();
    let options = LinalgExecutionOptions::new();
    let solve = DirectSolveOptions::new(RankPolicy::Exact);
    for op in [
        StructuredOperation::Normal,
        StructuredOperation::Transpose,
        StructuredOperation::Adjoint,
    ] {
        for layout in [BlasLayout::RowMajor, BlasLayout::ColumnMajor] {
            for family in 0..3 {
                let a = if family == 1 { &spd } else { &general };
                let a = match op {
                    StructuredOperation::Normal => a.view(),
                    StructuredOperation::Transpose => a.view().transpose(),
                    StructuredOperation::Adjoint => a.view().adjoint(),
                };

                let provider = layout_provider(layout);
                let mut actual = Matrix::zeros(n, 3).unwrap();
                let mut expected = Matrix::zeros(n, 3).unwrap();
                for (output, mut context) in [
                    (&mut expected, LinalgExecutionContext::sequential()),
                    (&mut actual, provider.context()),
                ] {
                    let mut workspace = LinalgWorkspace::new();
                    let report = match family {
                        0 => {
                            let mut factor = LuDecomposition::empty();
                            LuDecomposition::factor_into(
                                a,
                                &options,
                                &mut factor,
                                &mut workspace,
                                context.reborrow(),
                            )
                            .unwrap();
                            factor
                                .solve_matrix_into(
                                    &b,
                                    &solve,
                                    output.view_mut(),
                                    &mut workspace,
                                    context,
                                )
                                .unwrap()
                        }
                        1 => {
                            let mut factor = CholeskyDecomposition::empty();
                            CholeskyDecomposition::factor_into(
                                a,
                                scioxide_linalg::HermitianInput::Exact,
                                scioxide_linalg::CholeskyPivoting::None,
                                &options,
                                &mut factor,
                                &mut workspace,
                                context.reborrow(),
                            )
                            .unwrap();
                            factor
                                .solve_matrix_into(
                                    &b,
                                    &solve,
                                    output.view_mut(),
                                    &mut workspace,
                                    context,
                                )
                                .unwrap()
                        }
                        _ => {
                            let mut factor = HouseholderDecomposition::empty();
                            HouseholderDecomposition::factor_into(
                                a,
                                HouseholderOrientation::Qr,
                                scioxide_linalg::HouseholderPivoting::Column,
                                &options,
                                &mut factor,
                                &mut workspace,
                                context.reborrow(),
                            )
                            .unwrap();
                            let expected_q = factor.apply_q(&b, op).unwrap();
                            let mut actual_q = Matrix::zeros(n, 3).unwrap();
                            factor
                                .apply_q_into(
                                    &b,
                                    op,
                                    &options,
                                    actual_q.view_mut(),
                                    &mut workspace,
                                    context.reborrow(),
                                )
                                .unwrap();
                            close(actual_q.as_slice(), expected_q.as_slice(), n);
                            // A solve reservation must cover TRSM independently of
                            // capacity retained by factorization or Q application.
                            let request =
                                scioxide_linalg::HouseholderWorkspaceRequest::solve_matrix(
                                    n,
                                    n,
                                    HouseholderOrientation::Qr,
                                    b.cols(),
                                    &solve.execution_options(),
                                )
                                .for_context(&context);
                            let mut solve_workspace =
                                HouseholderDecomposition::<S>::workspace(&request).unwrap();
                            factor
                                .solve_matrix_into(
                                    &b,
                                    &solve,
                                    output.view_mut(),
                                    &mut solve_workspace,
                                    context,
                                )
                                .unwrap()
                        }
                    };
                    assert_eq!(
                        report.termination(),
                        scioxide_linalg::DirectSolveTermination::Accepted
                    );
                }
                close(actual.as_slice(), expected.as_slice(), n);
                for row in 0..n {
                    for col in 0..3 {
                        let value = (0..n).fold(S::ZERO, |sum, k| {
                            sum + a.get(row, k).unwrap() * actual.as_slice()[k * 3 + col]
                        });
                        close(&[value], &[b.as_slice()[row * 3 + col]], n);
                    }
                }
                assert_routed(&provider, BlasOperation::Trsm);
                if family >= 2 {
                    assert_routed(&provider, BlasOperation::Trmm);
                }
            }
        }
    }
}

#[test]
fn non_identity_factor_transposed_systems() {
    factor_solve_regressions::<f64>();
    factor_solve_regressions::<Complex64>();
}

#[test]
fn conjugated_structured_operands_obey_fallback_before_invocation() {
    let caps = BlasCapabilities::empty(BlasIntegerWidth::Lp64)
        .with_layout(BlasLayout::RowMajor)
        .with_transform(BlasTransform::None)
        .with_transform(BlasTransform::Transpose)
        .with_transform(BlasTransform::Adjoint);
    let caps = [
        BlasOperation::Symv,
        BlasOperation::Hemv,
        BlasOperation::Trsv,
    ]
    .into_iter()
    .filter(|o| o.supports_scalar(ScalarKind::Complex64))
    .fold(caps, |c, o| {
        c.with_operation(ScalarKind::Complex64, o, BlasUlp::new(4))
            .unwrap()
    });
    let provider = CblasProvider::with_capabilities(caps);
    let data = [
        Complex64::new(2.0, 0.0),
        Complex64::new(71.0, 4.0),
        Complex64::new(1.0, 2.0),
        Complex64::new(3.0, 0.0),
    ];
    let a = MatrixView::from_slice(2, 2, &data).unwrap().conjugate();
    let x = [Complex64::new(1.0, 0.5), Complex64::new(-0.3, 1.0)];
    for operation in 0..3 {
        let invoke = |out: &mut [Complex64], context: LinalgExecutionContext<'_, '_>| {
            let mut workspace = LinalgWorkspace::new();
            match operation {
                0 => a.symmetric_mul_vec_in_place(
                    Triangle::Lower,
                    Complex64::ONE,
                    VectorView::from_slice(&x),
                    Complex64::ZERO,
                    &LinalgExecutionOptions::new(),
                    VectorViewMut::from_slice(out),
                    &mut workspace,
                    context,
                ),
                1 => a.hermitian_mul_vec_in_place(
                    Triangle::Lower,
                    Complex64::ONE,
                    VectorView::from_slice(&x),
                    Complex64::ZERO,
                    &LinalgExecutionOptions::new(),
                    VectorViewMut::from_slice(out),
                    &mut workspace,
                    context,
                ),
                _ => {
                    let plan =
                        TriangularPlan::from_dense(a, Triangle::Lower, DiagonalKind::Explicit)
                            .unwrap();
                    plan.solve_into(
                        StructuredOperation::Normal,
                        VectorView::from_slice(&x),
                        &DirectSolveOptions::new(RankPolicy::Exact),
                        VectorViewMut::from_slice(out),
                        &mut workspace,
                        context,
                    )
                    .map(|_| ())
                }
            }
        };
        let initial = [Complex64::new(17.0, 8.0); 2];
        let mut expected = initial;
        invoke(&mut expected, LinalgExecutionContext::sequential()).unwrap();
        for fallback in [BlasFallbackPolicy::Reference, BlasFallbackPolicy::Error] {
            let mut out = initial;
            let result = invoke(
                &mut out,
                LinalgExecutionContext::sequential()
                    .with_blas(BlasDispatch::new(&provider, fallback)),
            );
            match fallback {
                BlasFallbackPolicy::Reference => {
                    result.unwrap();
                    close(&out, &expected, 2);
                }
                BlasFallbackPolicy::Error => {
                    assert!(matches!(
                        result.unwrap_err().view(),
                        scioxide_core::SciOxideErrorView::CapabilityUnavailable { .. }
                    ));
                    assert_eq!(out, initial);
                }
            }
            assert_eq!(provider.calls.load(Ordering::Relaxed), 0);
        }
    }
}

fn empty_extents<S: NumericScalar>() {
    let provider = CblasProvider::new();
    let options = LinalgExecutionOptions::new();
    for (m, k, n) in [(0, 3, 4), (3, 0, 4), (3, 4, 0), (0, 0, 0)] {
        let a = Matrix::<S>::zeros(m, k).unwrap();
        let b = Matrix::<S>::zeros(k, n).unwrap();
        let mut out = vec![S::ONE; m * n];
        a.view()
            .mul_mat_into(
                b.view(),
                &options,
                MatrixViewMut::from_slice(m, n, &mut out).unwrap(),
                &mut LinalgWorkspace::new(),
                provider.context(),
            )
            .unwrap();
        assert!(out.iter().all(|&x| x == S::ZERO));
    }
    for triangle in [Triangle::Lower, Triangle::Upper] {
        for n in [0, 3] {
            for beta in [0.0, 1.0, 2.0] {
                for transposed in [false, true] {
                    for hermitian in [false, true] {
                        let a = Matrix::<S>::zeros(
                            if transposed { 0 } else { n },
                            if transposed { n } else { 0 },
                        )
                        .unwrap();
                        let op = if !transposed {
                            StructuredOperation::Normal
                        } else if hermitian {
                            StructuredOperation::Adjoint
                        } else {
                            StructuredOperation::Transpose
                        };
                        for rank_two in [false, true] {
                            let mut expected = vec![scalar::<S>(2.0, 0.0); n * n];
                            if beta == 0.0 {
                                expected.fill(scalar(f64::NAN, 0.0));
                            }
                            let mut actual = expected.clone();
                            for (out, context) in [
                                (&mut expected, LinalgExecutionContext::sequential()),
                                (&mut actual, provider.context()),
                            ] {
                                let output = MatrixViewMut::from_slice(n, n, out).unwrap();
                                let mut w = LinalgWorkspace::new();
                                match (hermitian, rank_two) {
                                    (false, false) => {
                                        scioxide_linalg::MatrixViewMut::symmetric_rank_k_in_place(
                                            a.view(),
                                            triangle,
                                            op,
                                            S::ONE,
                                            scalar(beta, 0.0),
                                            &options,
                                            output,
                                            &mut w,
                                            context,
                                        )
                                    }
                                    (true, false) => {
                                        scioxide_linalg::MatrixViewMut::hermitian_rank_k_in_place(
                                            a.view(),
                                            triangle,
                                            op,
                                            S::ONE,
                                            scalar(beta, 0.0),
                                            &options,
                                            output,
                                            &mut w,
                                            context,
                                        )
                                    }
                                    (false, true) => {
                                        scioxide_linalg::MatrixViewMut::symmetric_rank_two_k_in_place(
                                            a.view(),
                                            a.view(),
                                            triangle,
                                            op,
                                            S::ONE,
                                            scalar(beta, 0.0),
                                            &options,
                                            output,
                                            &mut w,
                                            context,
                                        )
                                    }
                                    (true, true) => {
                                        scioxide_linalg::MatrixViewMut::hermitian_rank_two_k_in_place(
                                            a.view(),
                                            a.view(),
                                            triangle,
                                            op,
                                            S::ONE,
                                            scalar(beta, 0.0),
                                            &options,
                                            output,
                                            &mut w,
                                            context,
                                        )
                                    }
                                }
                                .unwrap();
                            }
                            close(&actual, &expected, 1);
                        }
                    }
                }
            }
        }
    }
    assert_eq!(provider.calls.load(Ordering::Relaxed), 0);
}

#[test]
fn empty_products_and_rank_updates_never_call_cblas() {
    empty_extents::<f32>();
    empty_extents::<f64>();
    empty_extents::<Complex32>();
    empty_extents::<Complex64>();
}

#[test]
fn rank_k_commits_only_validated_triangle_and_preserves_padding() {
    let a = Matrix::from_vec(3, 2, vec![1.0_f64, 2.0, 3.0, -0.5, 0.7, 1.1]).unwrap();
    for layout in [BlasLayout::RowMajor, BlasLayout::ColumnMajor] {
        for triangle in [Triangle::Lower, Triangle::Upper] {
            let provider = layout_provider(layout);
            provider.poison_unselected.store(true, Ordering::Relaxed);
            let initial = vec![17.0; 15];
            let mut actual = initial.clone();
            let mut expected = initial.clone();
            let invoke = |out: &mut [f64], context: LinalgExecutionContext<'_, '_>| {
                scioxide_linalg::MatrixViewMut::symmetric_rank_k_in_place(
                    a.view(),
                    triangle,
                    StructuredOperation::Normal,
                    0.7,
                    0.3,
                    &LinalgExecutionOptions::new(),
                    MatrixViewMut::from_slice_with_strides(3, 3, out, 5, 1).unwrap(),
                    &mut LinalgWorkspace::new(),
                    context,
                )
            };
            invoke(&mut expected, LinalgExecutionContext::sequential()).unwrap();
            invoke(&mut actual, provider.context()).unwrap();
            close(&actual, &expected, 3);
            for row in 0..3 {
                for col in 0..5 {
                    let selected = col < 3
                        && match triangle {
                            Triangle::Lower => row >= col,
                            Triangle::Upper => row <= col,
                        };
                    if !selected {
                        assert_eq!(
                            actual[row * 5 + col].to_bits(),
                            initial[row * 5 + col].to_bits()
                        );
                    }
                }
            }
            provider.malformed_result.store(true, Ordering::Relaxed);
            let before = actual.clone();
            let error = invoke(&mut actual, provider.context()).unwrap_err();
            assert!(matches!(
                error.view(),
                scioxide_core::SciOxideErrorView::NonFinite {
                    origin: NonFiniteOrigin::Callback,
                    ..
                }
            ));
            assert_eq!(actual, before);
        }
    }
}

#[test]
fn invalid_cblas_vector_increments_decline_before_invocation() {
    let provider = CblasProvider::new();
    for stride in [0, i32::MAX as usize + 1] {
        let data = [2.0_f64];
        let x = VectorView::from_slice(&[3.0]);
        let y = VectorView::from_slice_with_stride(&data, 1, stride).unwrap();
        for fallback in [BlasFallbackPolicy::Reference, BlasFallbackPolicy::Error] {
            let mut out = [7.0];
            let result = MatrixViewMut::rank_one_update_in_place(
                0.5,
                x,
                y,
                &LinalgExecutionOptions::new(),
                MatrixViewMut::from_slice(1, 1, &mut out).unwrap(),
                &mut LinalgWorkspace::new(),
                LinalgExecutionContext::sequential()
                    .with_blas(BlasDispatch::new(&provider, fallback)),
            );
            if fallback == BlasFallbackPolicy::Reference {
                result.unwrap();
                assert_eq!(out, [10.0]);
            } else {
                assert!(matches!(
                    result.unwrap_err().view(),
                    scioxide_core::SciOxideErrorView::CapabilityUnavailable { .. }
                ));
                assert_eq!(out, [7.0]);
            }
            assert_eq!(provider.calls.load(Ordering::Relaxed), 0);
        }
    }
}

#[test]
fn complex_symmetric_level_two_uses_explicit_fallback() {
    let provider = CblasProvider::new();
    let a = Matrix::from_vec(
        2,
        2,
        vec![
            Complex64::new(2.0, 1.0),
            Complex64::ZERO,
            Complex64::new(1.0, -0.5),
            Complex64::new(3.0, 0.2),
        ],
    )
    .unwrap();
    let x = [Complex64::new(0.5, 0.7), Complex64::new(-0.3, 0.2)];
    let invoke = |out: &mut [Complex64], context: LinalgExecutionContext<'_, '_>| {
        a.view().symmetric_mul_vec_in_place(
            Triangle::Lower,
            Complex64::ONE,
            VectorView::from_slice(&x),
            Complex64::ZERO,
            &LinalgExecutionOptions::new(),
            VectorViewMut::from_slice(out),
            &mut LinalgWorkspace::new(),
            context,
        )
    };
    let mut expected = [Complex64::ZERO; 2];
    invoke(&mut expected, LinalgExecutionContext::sequential()).unwrap();
    let mut actual = [Complex64::ONE; 2];
    invoke(
        &mut actual,
        LinalgExecutionContext::sequential()
            .with_blas(BlasDispatch::new(&provider, BlasFallbackPolicy::Reference)),
    )
    .unwrap();
    close(&actual, &expected, 2);
    let before = actual;
    assert!(matches!(
        invoke(&mut actual, provider.context()).unwrap_err().view(),
        scioxide_core::SciOxideErrorView::CapabilityUnavailable { .. }
    ));
    assert_eq!(actual, before);
    assert_eq!(provider.calls.load(Ordering::Relaxed), 0);
}

fn rectangular_rank_updates<S: NumericScalar>() {
    let n = 3;
    for k in [1, 5] {
        let a = Matrix::from_vec(
            n,
            k,
            (0..n * k)
                .map(|i| scalar::<S>((i + 1) as f64 / 7.0, (i % 3) as f64 / 5.0))
                .collect(),
        )
        .unwrap();
        let b = Matrix::from_vec(
            n,
            k,
            (0..n * k)
                .map(|i| scalar::<S>((i + 2) as f64 / 9.0, -(i as f64) / 11.0))
                .collect(),
        )
        .unwrap();
        for layout in [BlasLayout::RowMajor, BlasLayout::ColumnMajor] {
            for triangle in [Triangle::Lower, Triangle::Upper] {
                for hermitian in [false, true] {
                    for transposed in [false, true] {
                        for two in [false, true] {
                            let provider = layout_provider(layout);
                            let op = if !transposed {
                                StructuredOperation::Normal
                            } else if hermitian {
                                StructuredOperation::Adjoint
                            } else {
                                StructuredOperation::Transpose
                            };
                            let (a, b) = match op {
                                StructuredOperation::Normal => (a.view(), b.view()),
                                StructuredOperation::Transpose => {
                                    (a.view().transpose(), b.view().transpose())
                                }
                                StructuredOperation::Adjoint => {
                                    (a.view().adjoint(), b.view().adjoint())
                                }
                            };
                            let mut actual = vec![scalar::<S>(0.2, 0.0); n * n];
                            let mut expected = actual.clone();
                            for (out, context) in [
                                (&mut expected, LinalgExecutionContext::sequential()),
                                (&mut actual, provider.context()),
                            ] {
                                let mut w = LinalgWorkspace::new();
                                let output = MatrixViewMut::from_slice(n, n, out).unwrap();
                                let options = LinalgExecutionOptions::new();
                                match (hermitian, two) {
                                    (false, false) => {
                                        scioxide_linalg::MatrixViewMut::symmetric_rank_k_in_place(
                                            a,
                                            triangle,
                                            op,
                                            scalar(0.7, 0.0),
                                            scalar(0.3, 0.0),
                                            &options,
                                            output,
                                            &mut w,
                                            context,
                                        )
                                    }
                                    (true, false) => {
                                        scioxide_linalg::MatrixViewMut::hermitian_rank_k_in_place(
                                            a,
                                            triangle,
                                            op,
                                            scalar(0.7, 0.0),
                                            scalar(0.3, 0.0),
                                            &options,
                                            output,
                                            &mut w,
                                            context,
                                        )
                                    }
                                    (false, true) => {
                                        scioxide_linalg::MatrixViewMut::symmetric_rank_two_k_in_place(
                                            a,
                                            b,
                                            triangle,
                                            op,
                                            scalar(0.7, 0.0),
                                            scalar(0.3, 0.0),
                                            &options,
                                            output,
                                            &mut w,
                                            context,
                                        )
                                    }
                                    (true, true) => {
                                        scioxide_linalg::MatrixViewMut::hermitian_rank_two_k_in_place(
                                            a,
                                            b,
                                            triangle,
                                            op,
                                            scalar(0.7, 0.2),
                                            scalar(0.3, 0.0),
                                            &options,
                                            output,
                                            &mut w,
                                            context,
                                        )
                                    }
                                }
                                .unwrap();
                            }
                            close(&actual, &expected, k);
                            assert_routed(
                                &provider,
                                match (hermitian && !S::IS_REAL, two) {
                                    (false, false) => BlasOperation::Syrk,
                                    (true, false) => BlasOperation::Herk,
                                    (false, true) => BlasOperation::Syr2k,
                                    (true, true) => BlasOperation::Her2k,
                                },
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn rectangular_rank_k_and_rank_two_k_use_cblas_transforms() {
    rectangular_rank_updates::<f32>();
    rectangular_rank_updates::<f64>();
    rectangular_rank_updates::<Complex32>();
    rectangular_rank_updates::<Complex64>();
}

#[test]
fn gemv_reencodes_conjugation_using_a_supported_cblas_layout() {
    let provider = CblasProvider::new();
    let data = [
        Complex64::new(1.0, 2.0),
        Complex64::new(0.5, -0.3),
        Complex64::new(-2.0, 1.0),
        Complex64::new(3.0, 0.2),
        Complex64::new(0.4, -0.7),
        Complex64::new(2.0, 0.6),
    ];
    let a = MatrixView::from_slice(2, 3, &data).unwrap().conjugate();
    let x = [
        Complex64::new(0.3, 0.7),
        Complex64::new(0.4, -0.5),
        Complex64::new(-0.2, 0.8),
    ];
    let mut actual = [Complex64::ZERO; 2];
    let mut expected = actual;
    for (out, context) in [
        (&mut expected, LinalgExecutionContext::sequential()),
        (&mut actual, provider.context()),
    ] {
        a.mul_vec_into(
            VectorView::from_slice(&x),
            &LinalgExecutionOptions::new(),
            VectorViewMut::from_slice(out),
            &mut LinalgWorkspace::new(),
            context,
        )
        .unwrap();
    }
    close(&actual, &expected, 3);
    assert_routed(&provider, BlasOperation::Gemv);
}

fn check_matrix_case<S: NumericScalar>(provider: &CblasProvider, op: BlasOperation) {
    provider.operations.store(0, Ordering::Relaxed);
    close(
        &matrix_case::<S>(op, provider.context()),
        &matrix_case::<S>(op, LinalgExecutionContext::sequential()),
        3,
    );
    assert_routed(provider, op);
}

fn vector_scalar_routes<S: NumericScalar>() {
    let provider = CblasProvider::new();
    for op in [
        BlasOperation::Copy,
        BlasOperation::Scale,
        BlasOperation::Axpy,
        BlasOperation::Dotu,
        BlasOperation::Dotc,
        BlasOperation::Norm2,
        BlasOperation::Asum,
        BlasOperation::Iamax,
    ] {
        provider.operations.store(0, Ordering::Relaxed);
        let x_data = [scalar::<S>(0.3, 0.5), scalar(-0.7, 0.2), scalar(0.8, -0.6)];
        let y_data = [scalar::<S>(0.4, -0.2), scalar(0.5, 0.7), scalar(-0.1, 0.3)];
        let invoke = |context: LinalgExecutionContext<'_, '_>| {
            let x = VectorView::from_slice(&x_data);
            let y = VectorView::from_slice(&y_data);
            let mut out = y_data.to_vec();
            let mut w = LinalgWorkspace::new();
            let options = LinalgExecutionOptions::new();
            match op {
                BlasOperation::Copy => x
                    .copy_into(
                        scioxide_linalg::CopyPolicy::RequireFinite,
                        &options,
                        VectorViewMut::from_slice(&mut out),
                        &mut w,
                        context,
                    )
                    .unwrap(),
                BlasOperation::Scale => VectorViewMut::scale_in_place(
                    scalar(0.7, 0.1),
                    &options,
                    VectorViewMut::from_slice(&mut out),
                    &mut w,
                    context,
                )
                .unwrap(),
                BlasOperation::Axpy => x
                    .axpy_into(
                        scalar(0.7, 0.1),
                        y,
                        &options,
                        VectorViewMut::from_slice(&mut out),
                        &mut w,
                        context,
                    )
                    .unwrap(),
                BlasOperation::Dotu => {
                    return vec![
                        x.dotu_scoped(y, &scioxide_linalg::LinalgExecutionOptions::new(), context)
                            .unwrap(),
                    ];
                }
                BlasOperation::Dotc => {
                    return vec![
                        x.dotc_left_scoped(
                            y,
                            &scioxide_linalg::LinalgExecutionOptions::new(),
                            context,
                        )
                        .unwrap(),
                    ];
                }
                BlasOperation::Norm2 => {
                    return vec![S::from_real(
                        x.norm2_scoped(&scioxide_linalg::LinalgExecutionOptions::new(), context)
                            .unwrap(),
                    )];
                }
                BlasOperation::Asum => {
                    return vec![S::from_real(
                        x.asum_scoped(&scioxide_linalg::LinalgExecutionOptions::new(), context)
                            .unwrap(),
                    )];
                }
                BlasOperation::Iamax => {
                    return vec![scalar(
                        x.iamax_scoped(&scioxide_linalg::LinalgExecutionOptions::new(), context)
                            .unwrap()
                            .unwrap() as f64,
                        0.0,
                    )];
                }
                _ => unreachable!(),
            }
            out
        };
        close(
            &invoke(provider.context()),
            &invoke(LinalgExecutionContext::sequential()),
            3,
        );
        assert_routed(&provider, op);
    }
}

fn real_rotation_routes<R: RealScalar + NumericScalar<Real = R>>() {
    let provider = CblasProvider::new();
    for op in [
        BlasOperation::Swap,
        BlasOperation::Rot,
        BlasOperation::Rotm,
        BlasOperation::Rotg,
        BlasOperation::Rotmg,
    ] {
        provider.operations.store(0, Ordering::Relaxed);
        let invoke = |mut context: LinalgExecutionContext<'_, '_>| {
            let mut x = vec![scalar::<R>(0.3, 0.0), scalar(-0.7, 0.0)];
            let mut y = vec![scalar::<R>(0.5, 0.0), scalar(0.8, 0.0)];
            let options = LinalgExecutionOptions::new();
            let mut w = LinalgWorkspace::new();
            let rotation = scioxide_linalg::givens_rotation_scoped(
                scalar::<R>(3.0, 0.0),
                scalar(4.0, 0.0),
                &scioxide_linalg::LinalgExecutionOptions::new(),
                context.reborrow(),
            )
            .unwrap();
            if op == BlasOperation::Rotg {
                return vec![
                    rotation.r(),
                    rotation.z(),
                    rotation.rotation().cosine(),
                    rotation.rotation().sine(),
                ];
            }
            let modified = scioxide_linalg::modified_givens_rotation_scoped(
                R::ONE,
                R::ONE,
                scalar(3.0, 0.0),
                scalar(4.0, 0.0),
                &scioxide_linalg::LinalgExecutionOptions::new(),
                context.reborrow(),
            )
            .unwrap();
            if op == BlasOperation::Rotmg {
                return vec![modified.d1(), modified.d2(), modified.x1()];
            }
            match op {
                BlasOperation::Swap => VectorViewMut::swap_in_place(
                    &options,
                    VectorViewMut::from_slice(&mut x),
                    VectorViewMut::from_slice(&mut y),
                    &mut w,
                    context,
                )
                .unwrap(),
                BlasOperation::Rot => VectorViewMut::rotate_in_place(
                    rotation.rotation(),
                    &options,
                    VectorViewMut::from_slice(&mut x),
                    VectorViewMut::from_slice(&mut y),
                    &mut w,
                    context,
                )
                .unwrap(),
                BlasOperation::Rotm => VectorViewMut::modified_rotate_in_place(
                    modified.rotation(),
                    &options,
                    VectorViewMut::from_slice(&mut x),
                    VectorViewMut::from_slice(&mut y),
                    &mut w,
                    context,
                )
                .unwrap(),
                _ => unreachable!(),
            }
            x.extend(y);
            x
        };
        close(
            &invoke(provider.context()),
            &invoke(LinalgExecutionContext::sequential()),
            3,
        );
        assert_routed(&provider, op);
    }
}

#[test]
fn level_one_routes_cover_every_applicable_scalar_entrypoint() {
    vector_scalar_routes::<f32>();
    vector_scalar_routes::<f64>();
    vector_scalar_routes::<Complex32>();
    vector_scalar_routes::<Complex64>();
    real_rotation_routes::<f32>();
    real_rotation_routes::<f64>();
}
