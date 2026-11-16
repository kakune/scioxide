#![cfg(feature = "blas")]
#![allow(clippy::unwrap_used)]

use core::num::NonZeroUsize;
use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use scioxide_core::{AlgebraicScalar, Triangle};

use scioxide_core::backend::ScalarKind;
use scioxide_core::{
    Capability, CapabilityError, Complex32, Complex64, ExecutionContext, ExecutionPlan,
    ExecutionPolicy, Executor, NonFiniteOrigin, NumericScalar, PartitionFn, RealScalar,
    SciOxideError, SequentialExecutor,
};
use scioxide_linalg::{
    BandedMatrixView, BlasCall, BlasCapabilities, BlasDispatch, BlasFallbackPolicy,
    BlasIntegerWidth, BlasLayout, BlasMatrix, BlasMatrixMut, BlasOperation, BlasProvider,
    BlasThreading, BlasTransform, BlasUlp, BlasVector, BlasVectorMut, CholeskyDecomposition,
    HouseholderDecomposition, HouseholderOrientation, LinalgExecutionContext,
    LinalgExecutionOptions, LinalgWorkspace, LuDecomposition, Matrix, MatrixView,
    StructuredOperation, Vector, VectorView, VectorViewMut,
};

#[derive(Clone, Copy)]
enum Operand<'a, S> {
    Dense {
        matrix: BlasMatrix<'a, S>,
        triangle: Triangle,
    },
    Band {
        matrix: scioxide_linalg::BlasBandMatrix<'a, S>,
        triangle: Option<Triangle>,
    },
    Packed(scioxide_linalg::BlasPackedMatrix<'a, S>),
}
enum OperandMut<'a, S> {
    Dense(BlasMatrixMut<'a, S>),
    Packed(scioxide_linalg::BlasPackedMatrixMut<'a, S>),
}

struct CblasProvider {
    calls: AtomicUsize,
    operations: std::sync::atomic::AtomicU64,
    workers: AtomicUsize,
    fail_after_write: AtomicBool,
    malformed_result: AtomicBool,
    poison_unselected: AtomicBool,
    poison_upper_gemm: AtomicBool,
    capabilities: BlasCapabilities,
}

impl CblasProvider {
    fn new() -> Self {
        let capabilities = conformance::ALL_OPERATIONS.into_iter().fold(
            BlasCapabilities::empty(BlasIntegerWidth::Lp64)
                .with_layout(BlasLayout::RowMajor)
                .with_layout(BlasLayout::ColumnMajor)
                .with_transform(BlasTransform::None)
                .with_transform(BlasTransform::Transpose)
                .with_transform(BlasTransform::Conjugate)
                .with_transform(BlasTransform::Adjoint)
                .with_threading(BlasThreading::Budgeted),
            |capabilities, operation| {
                [
                    ScalarKind::F32,
                    ScalarKind::F64,
                    ScalarKind::Complex32,
                    ScalarKind::Complex64,
                ]
                .into_iter()
                .filter(|&scalar| operation.supports_scalar(scalar))
                .fold(capabilities, |caps, scalar| {
                    caps.with_operation(scalar, operation, BlasUlp::new(4))
                        .unwrap()
                })
            },
        );
        Self {
            calls: AtomicUsize::new(0),
            operations: std::sync::atomic::AtomicU64::new(0),
            workers: AtomicUsize::new(0),
            fail_after_write: AtomicBool::new(false),
            malformed_result: AtomicBool::new(false),
            poison_unselected: AtomicBool::new(false),
            poison_upper_gemm: AtomicBool::new(false),
            capabilities,
        }
    }

    fn context(&self) -> LinalgExecutionContext<'_, '_> {
        LinalgExecutionContext::sequential()
            .with_blas(BlasDispatch::new(self, BlasFallbackPolicy::Error))
    }

    fn with_capabilities(capabilities: BlasCapabilities) -> Self {
        Self {
            capabilities,
            ..Self::new()
        }
    }

    fn execute<S: NumericScalar>(&self, call: BlasCall<'_, S>) -> Result<(), CapabilityError> {
        conformance::validate_call(&call, &self.capabilities);
        self.operations
            .fetch_or(operation_bit(&call), Ordering::Relaxed);
        macro_rules! matvec {
            ($operation:expr, $transform:expr, $alpha:expr, $a:expr, $x:expr, $beta:expr, $y:expr) => {{
                let operation = $operation;
                let transform = $transform;
                let alpha = $alpha;
                let a = $a;
                let x = $x;
                let beta = $beta;
                let mut y = $y;
                let (rows, cols, triangle) = structured_shape(a);
                let general = operation == BlasOperation::Gbmv;
                let hermitian = matches!(
                    operation,
                    BlasOperation::Hemv | BlasOperation::Hbmv | BlasOperation::Hpmv
                );
                let raw = |row, col| {
                    if general || triangle_contains(triangle, row, col) {
                        let v = structured_get(a, row, col);
                        if hermitian && row == col {
                            S::from_real(v.real_component())
                        } else {
                            v
                        }
                    } else {
                        let v = structured_get(a, col, row);
                        if hermitian { v.conj() } else { v }
                    }
                };
                let transposed =
                    matches!(transform, BlasTransform::Transpose | BlasTransform::Adjoint);
                let (m, n) = if transposed {
                    (cols, rows)
                } else {
                    (rows, cols)
                };
                for row in 0..m {
                    let sum = (0..n).fold(S::ZERO, |sum, col| {
                        let v = if transposed {
                            raw(col, row)
                        } else {
                            raw(row, col)
                        };
                        let v = if matches!(
                            transform,
                            BlasTransform::Adjoint | BlasTransform::Conjugate
                        ) {
                            v.conj()
                        } else {
                            v
                        };
                        sum + v * vector_get(x, col)
                    });
                    let prior = if beta == S::ZERO {
                        S::ZERO
                    } else {
                        beta * vector_mut_get(&y, row)
                    };
                    vector_set(&mut y, row, alpha * sum + prior);
                }
                if self.fail_after_write.load(Ordering::Relaxed) && m != 0 {
                    return Err(CapabilityError::new("injected extended provider failure"));
                }
            }};
        }
        macro_rules! triangular_vector {
            ($transform:expr, $diagonal:expr, $a:expr, $x:expr) => {{
                let transform = $transform;
                let diagonal = $diagonal;
                let a = $a;
                let mut x = $x;
                let (n, _, triangle) = structured_shape(a);
                let transpose =
                    matches!(transform, BlasTransform::Transpose | BlasTransform::Adjoint);
                let coefficient = |row, col| {
                    let (r, c) = if transpose { (col, row) } else { (row, col) };
                    let v = if r == c && diagonal == scioxide_core::DiagonalKind::Unit {
                        S::ONE
                    } else if triangle_contains(triangle, r, c) {
                        structured_get(a, r, c)
                    } else {
                        S::ZERO
                    };
                    if matches!(transform, BlasTransform::Adjoint | BlasTransform::Conjugate) {
                        v.conj()
                    } else {
                        v
                    }
                };
                let old = (0..n).map(|i| vector_mut_get(&x, i)).collect::<Vec<_>>();
                let mut result = old.clone();
                let lower = (triangle == scioxide_core::Triangle::Lower) != transpose;
                for step in 0..n {
                    let row = if lower { step } else { n - 1 - step };
                    let value = (0..n)
                        .filter(|&col| if lower { col < row } else { col > row })
                        .fold(old[row], |v, col| v - coefficient(row, col) * result[col]);
                    result[row] = value / coefficient(row, row);
                }
                for (index, value) in result.into_iter().enumerate() {
                    vector_set(&mut x, index, value);
                }
            }};
        }
        macro_rules! rank_update {
            ($operation:expr, $triangle:expr, $alpha:expr, $x:expr, $y:expr, $a:expr) => {{
                let operation = $operation;
                let triangle = $triangle;
                let alpha = $alpha;
                let x = $x;
                let y = $y;
                let mut a = $a;
                let (rows, cols) = match &a {
                    OperandMut::Dense(a) => (a.rows(), a.cols()),
                    OperandMut::Packed(a) => (a.order(), a.order()),
                };
                let general = matches!(operation, BlasOperation::Geru | BlasOperation::Gerc);
                let hermitian = matches!(
                    operation,
                    BlasOperation::Her
                        | BlasOperation::Her2
                        | BlasOperation::Hpr
                        | BlasOperation::Hpr2
                );
                for row in 0..rows {
                    for col in 0..cols {
                        if !general && !triangle_contains(triangle, row, col) {
                            continue;
                        }
                        let xr = vector_get(x, row);
                        let xc = vector_get(x, col.min(x.len() - 1));
                        let product = if let Some(y) = y {
                            let yr = vector_get(y, row.min(y.len() - 1));
                            let yc = vector_get(y, col);
                            if general {
                                alpha
                                    * xr
                                    * if operation == BlasOperation::Gerc {
                                        yc.conj()
                                    } else {
                                        yc
                                    }
                            } else if hermitian {
                                alpha * xr * yc.conj() + alpha.conj() * yr * xc.conj()
                            } else {
                                alpha * (xr * yc + yr * xc)
                            }
                        } else {
                            alpha * xr * if hermitian { xc.conj() } else { xc }
                        };
                        let value = match &a {
                            OperandMut::Dense(a) => matrix_mut_get(a, row, col),
                            OperandMut::Packed(a) => {
                                a.values()[packed_index(a.order(), triangle, row, col)]
                            }
                        } + product;
                        let value = if hermitian && row == col {
                            S::from_real(value.real_component())
                        } else {
                            value
                        };
                        match &mut a {
                            OperandMut::Dense(a) => matrix_set(a, row, col, value),
                            OperandMut::Packed(a) => {
                                let index = packed_index(a.order(), triangle, row, col);
                                a.values_mut()[index] = value;
                            }
                        }
                    }
                }
            }};
        }
        match call {
            BlasCall::Copy { x, mut y } => {
                (0..x.len()).for_each(|index| {
                    let value = vector_get(x, index);
                    vector_set(&mut y, index, value);
                });
            }
            BlasCall::Scale { alpha, mut x } => {
                (0..x.len()).for_each(|index| {
                    let value = vector_mut_get(&x, index);
                    vector_set(&mut x, index, alpha * value);
                });
            }
            BlasCall::Axpy { alpha, x, mut y } => {
                (0..x.len()).for_each(|index| {
                    let value = alpha * vector_get(x, index) + vector_mut_get(&y, index);
                    vector_set(&mut y, index, value);
                });
            }
            BlasCall::Dotu { x, y, result } => {
                let conjugate_left = false;

                *result = (0..x.len()).fold(S::ZERO, |sum, index| {
                    let left = vector_get(x, index);
                    sum + if conjugate_left { left.conj() } else { left } * vector_get(y, index)
                });
            }
            BlasCall::Dotc { x, y, result } => {
                let conjugate_left = true;

                *result = (0..x.len()).fold(S::ZERO, |sum, index| {
                    let left = vector_get(x, index);
                    sum + if conjugate_left { left.conj() } else { left } * vector_get(y, index)
                });
            }
            BlasCall::Norm2 { x, result } => {
                let sum = (0..x.len()).fold(S::Real::ZERO, |sum, index| {
                    sum + vector_get(x, index).norm_sqr()
                });
                *result = sum.sqrt();
            }
            BlasCall::Asum { x, result } => {
                let total = (0..x.len()).fold(S::Real::ZERO, |sum, index| {
                    let value = vector_get(x, index);
                    sum + value.real_component().abs() + value.imaginary_component().abs()
                });
                *result = total;
            }
            BlasCall::Iamax { x, result } => {
                let mut best = S::Real::ZERO;
                for index in 0..x.len() {
                    let value = vector_get(x, index);
                    let magnitude =
                        value.real_component().abs() + value.imaginary_component().abs();
                    if index == 0 || magnitude > best {
                        best = magnitude;
                        *result = index;
                    }
                }
            }
            BlasCall::Gemv {
                transform,
                alpha,
                a,
                x,
                beta,
                mut y,
            } => {
                let (rows, cols) = matrix_logical_shape(a, transform);
                (0..rows).for_each(|row| {
                    let product = (0..cols).fold(S::ZERO, |sum, col| {
                        sum + op_get(a, transform, row, col) * vector_get(x, col)
                    });
                    let prior = if beta == S::ZERO {
                        S::ZERO
                    } else {
                        beta * vector_mut_get(&y, row)
                    };
                    let value = alpha * product + prior;
                    vector_set(&mut y, row, value);
                });
                if self.fail_after_write.load(Ordering::Relaxed) {
                    return Err(CapabilityError::new("injected GEMV provider failure"));
                }
            }
            BlasCall::Gemm {
                left_transform,
                right_transform,
                alpha,
                a,
                b,
                beta,
                mut c,
            } => {
                let (rows, shared) = matrix_logical_shape(a, left_transform);
                let (_, cols) = matrix_logical_shape(b, right_transform);
                (0..rows).for_each(|row| {
                    (0..cols).for_each(|col| {
                        let product = (0..shared).fold(S::ZERO, |sum, inner| {
                            sum + op_get(a, left_transform, row, inner)
                                * op_get(b, right_transform, inner, col)
                        });
                        let prior = if beta == S::ZERO {
                            S::ZERO
                        } else {
                            beta * matrix_mut_get(&c, row, col)
                        };
                        matrix_set(&mut c, row, col, alpha * product + prior);
                    });
                });
                // Square GEMMs may carry a lower publication mask that the provider
                // cannot see; garbage above the diagonal must then be ignored.
                if self.poison_upper_gemm.load(Ordering::Relaxed) && rows == cols {
                    for row in 0..rows {
                        for col in row + 1..cols {
                            matrix_set(&mut c, row, col, S::from_real(S::Real::NAN));
                        }
                    }
                }
                if self.fail_after_write.load(Ordering::Relaxed) && rows != 0 && cols != 0 {
                    matrix_set(&mut c, 0, 0, S::ZERO);
                    return Err(CapabilityError::new("injected provider failure"));
                }
            }
            BlasCall::Syrk {
                transform,
                triangle,
                alpha,
                a,
                beta,
                mut c,
            } => {
                let hermitian = false;

                let (order, shared) = matrix_logical_shape(a, transform);
                (0..order).for_each(|row| {
                    (0..order)
                        .filter(|&col| triangle_contains(triangle, row, col))
                        .for_each(|col| {
                            let product = (0..shared).fold(S::ZERO, |sum, inner| {
                                let right = op_get(a, transform, col, inner);
                                sum + op_get(a, transform, row, inner)
                                    * if hermitian { right.conj() } else { right }
                            });
                            let prior = if beta == S::ZERO {
                                S::ZERO
                            } else {
                                beta * matrix_mut_get(&c, row, col)
                            };
                            matrix_set(&mut c, row, col, alpha * product + prior);
                        });
                });
                if self.poison_unselected.load(Ordering::Relaxed) {
                    for row in 0..order {
                        for col in 0..order {
                            if !triangle_contains(triangle, row, col) {
                                matrix_set(&mut c, row, col, S::from_real(S::Real::NAN));
                            }
                        }
                    }
                }
                if self.malformed_result.load(Ordering::Relaxed) && !hermitian && order != 0 {
                    matrix_set(&mut c, 0, 0, S::from_real(S::Real::NAN));
                }
                if self.malformed_result.load(Ordering::Relaxed) && hermitian && order != 0 {
                    matrix_set(&mut c, 0, 0, S::from_components(S::Real::ONE, S::Real::ONE));
                }
            }
            BlasCall::Herk {
                transform,
                triangle,
                alpha,
                a,
                beta,
                mut c,
            } => {
                let hermitian = true;
                let alpha = S::from_real(alpha);
                let beta = S::from_real(beta);

                let (order, shared) = matrix_logical_shape(a, transform);
                (0..order).for_each(|row| {
                    (0..order)
                        .filter(|&col| triangle_contains(triangle, row, col))
                        .for_each(|col| {
                            let product = (0..shared).fold(S::ZERO, |sum, inner| {
                                let right = op_get(a, transform, col, inner);
                                sum + op_get(a, transform, row, inner)
                                    * if hermitian { right.conj() } else { right }
                            });
                            let prior = if beta == S::ZERO {
                                S::ZERO
                            } else {
                                beta * matrix_mut_get(&c, row, col)
                            };
                            matrix_set(&mut c, row, col, alpha * product + prior);
                        });
                });
                if self.poison_unselected.load(Ordering::Relaxed) {
                    for row in 0..order {
                        for col in 0..order {
                            if !triangle_contains(triangle, row, col) {
                                matrix_set(&mut c, row, col, S::from_real(S::Real::NAN));
                            }
                        }
                    }
                }
                if self.malformed_result.load(Ordering::Relaxed) && !hermitian && order != 0 {
                    matrix_set(&mut c, 0, 0, S::from_real(S::Real::NAN));
                }
                if self.malformed_result.load(Ordering::Relaxed) && hermitian && order != 0 {
                    matrix_set(&mut c, 0, 0, S::from_components(S::Real::ONE, S::Real::ONE));
                }
            }
            BlasCall::Trmm {
                transform,
                side,
                triangle,
                diagonal,
                alpha,
                a,
                mut b,
            } => {
                let solve = false;

                apply_triangular(solve, side, triangle, diagonal, transform, alpha, a, &mut b);
            }
            BlasCall::Trsm {
                transform,
                side,
                triangle,
                diagonal,
                alpha,
                a,
                mut b,
            } => {
                let solve = true;

                apply_triangular(solve, side, triangle, diagonal, transform, alpha, a, &mut b);
            }
            BlasCall::Swap { mut x, mut y } => {
                for index in 0..x.len() {
                    let left = vector_mut_get(&x, index);
                    let right = vector_mut_get(&y, index);
                    vector_set(&mut x, index, right);
                    vector_set(&mut y, index, left);
                }
            }
            BlasCall::Rot {
                mut x,
                mut y,
                cosine,
                sine,
            } => {
                let operation = BlasOperation::Rot;
                let parameters: Option<[S::Real; 5]> = None;

                for index in 0..x.len() {
                    let old_x = vector_mut_get(&x, index);
                    let old_y = vector_mut_get(&y, index);
                    let (next_x, next_y) = match operation {
                        BlasOperation::Swap => (old_y, old_x),
                        BlasOperation::Rot => {
                            (cosine * old_x + sine * old_y, cosine * old_y - sine * old_x)
                        }
                        BlasOperation::Rotm => {
                            let p = parameters
                                .ok_or_else(|| CapabilityError::new("missing ROTM parameters"))?;
                            let (h11, h21, h12, h22) = if p[0] == -S::Real::ONE {
                                (p[1], p[2], p[3], p[4])
                            } else if p[0] == S::Real::ZERO {
                                (S::Real::ONE, p[2], p[3], S::Real::ONE)
                            } else if p[0] == S::Real::ONE {
                                (p[1], -S::Real::ONE, S::Real::ONE, p[4])
                            } else {
                                (S::Real::ONE, S::Real::ZERO, S::Real::ZERO, S::Real::ONE)
                            };
                            (h11 * old_x + h12 * old_y, h21 * old_x + h22 * old_y)
                        }
                        _ => return Err(CapabilityError::new("unsupported pair operation")),
                    };
                    vector_set(&mut x, index, next_x);
                    vector_set(&mut y, index, next_y);
                }
            }
            BlasCall::Rotm {
                mut x,
                mut y,
                parameters,
            } => {
                let operation = BlasOperation::Rotm;
                let parameters = Some(parameters);
                let (cosine, sine) = (S::Real::ZERO, S::Real::ZERO);

                for index in 0..x.len() {
                    let old_x = vector_mut_get(&x, index);
                    let old_y = vector_mut_get(&y, index);
                    let (next_x, next_y) = match operation {
                        BlasOperation::Swap => (old_y, old_x),
                        BlasOperation::Rot => {
                            (cosine * old_x + sine * old_y, cosine * old_y - sine * old_x)
                        }
                        BlasOperation::Rotm => {
                            let p = parameters
                                .ok_or_else(|| CapabilityError::new("missing ROTM parameters"))?;
                            let (h11, h21, h12, h22) = if p[0] == -S::Real::ONE {
                                (p[1], p[2], p[3], p[4])
                            } else if p[0] == S::Real::ZERO {
                                (S::Real::ONE, p[2], p[3], S::Real::ONE)
                            } else if p[0] == S::Real::ONE {
                                (p[1], -S::Real::ONE, S::Real::ONE, p[4])
                            } else {
                                (S::Real::ONE, S::Real::ZERO, S::Real::ZERO, S::Real::ONE)
                            };
                            (h11 * old_x + h12 * old_y, h21 * old_x + h22 * old_y)
                        }
                        _ => return Err(CapabilityError::new("unsupported pair operation")),
                    };
                    vector_set(&mut x, index, next_x);
                    vector_set(&mut y, index, next_y);
                }
            }
            BlasCall::Rotg { a, b, cosine, sine } => {
                let first = a.real_component();
                let second = b.real_component();
                let magnitude = first.hypot(second);
                let r = magnitude.copysign(if first.abs() > second.abs() {
                    first
                } else {
                    second
                });
                let c = if r == S::Real::ZERO {
                    S::Real::ONE
                } else {
                    first / r
                };
                let s = if r == S::Real::ZERO {
                    S::Real::ZERO
                } else {
                    second / r
                };
                let z = if first.abs() > second.abs() {
                    s
                } else if c != S::Real::ZERO {
                    S::Real::ONE / c
                } else {
                    S::Real::ONE
                };
                *a = r;
                *b = z;
                *cosine = c;
                *sine = s;
            }
            BlasCall::Rotmg {
                d1,
                d2,
                x1,
                y1,
                parameters,
            } => {
                let mut first = d1.real_component();
                let mut second = d2.real_component();
                let mut x = x1.real_component();
                let p = cblas_rotmg(&mut first, &mut second, &mut x, y1.real_component());
                *d1 = first;
                *d2 = second;
                *x1 = x;
                *parameters = p;
            }
            BlasCall::WideDot { x, y, result } => {
                *result = (0..x.len()).fold(0.0_f64, |sum, index| {
                    sum + vector_get(x, index).real_component().into_f64()
                        * vector_get(y, index).real_component().into_f64()
                });
            }
            BlasCall::OffsetDot {
                x,
                y,
                offset,
                result,
            } => {
                let wide = (0..x.len()).fold(offset.real_component().into_f64(), |sum, index| {
                    sum + vector_get(x, index).real_component().into_f64()
                        * vector_get(y, index).real_component().into_f64()
                });
                *result = wide as f32;
            }
            BlasCall::Gbmv {
                transform,
                alpha,
                a,
                x,
                beta,
                y,
            } => {
                let operation = BlasOperation::Gbmv;
                let a = Operand::Band {
                    matrix: a,
                    triangle: None,
                };

                matvec!(operation, transform, alpha, a, x, beta, y);
            }
            BlasCall::Symv {
                triangle,
                alpha,
                a,
                x,
                beta,
                y,
            } => {
                let operation = BlasOperation::Symv;
                let a = Operand::Dense {
                    matrix: a,
                    triangle,
                };
                let transform = BlasTransform::None;

                matvec!(operation, transform, alpha, a, x, beta, y);
            }
            BlasCall::Hemv {
                triangle,
                alpha,
                a,
                x,
                beta,
                y,
            } => {
                let operation = BlasOperation::Hemv;
                let a = Operand::Dense {
                    matrix: a,
                    triangle,
                };
                let transform = BlasTransform::None;

                matvec!(operation, transform, alpha, a, x, beta, y);
            }
            BlasCall::Sbmv {
                triangle,
                alpha,
                a,
                x,
                beta,
                y,
            } => {
                let operation = BlasOperation::Sbmv;
                let a = Operand::Band {
                    matrix: a,
                    triangle: Some(triangle),
                };
                let transform = BlasTransform::None;

                matvec!(operation, transform, alpha, a, x, beta, y);
            }
            BlasCall::Hbmv {
                triangle,
                alpha,
                a,
                x,
                beta,
                y,
            } => {
                let operation = BlasOperation::Hbmv;
                let a = Operand::Band {
                    matrix: a,
                    triangle: Some(triangle),
                };
                let transform = BlasTransform::None;

                matvec!(operation, transform, alpha, a, x, beta, y);
            }
            BlasCall::Spmv {
                alpha,
                a,
                x,
                beta,
                y,
            } => {
                let operation = BlasOperation::Spmv;
                let a = Operand::Packed(a);
                let transform = BlasTransform::None;

                matvec!(operation, transform, alpha, a, x, beta, y);
            }
            BlasCall::Hpmv {
                alpha,
                a,
                x,
                beta,
                y,
            } => {
                let operation = BlasOperation::Hpmv;
                let a = Operand::Packed(a);
                let transform = BlasTransform::None;

                matvec!(operation, transform, alpha, a, x, beta, y);
            }
            BlasCall::Trsv {
                triangle,
                transform,
                diagonal,
                a,
                x,
            } => {
                let a = Operand::Dense {
                    matrix: a,
                    triangle,
                };

                triangular_vector!(transform, diagonal, a, x);
            }
            BlasCall::Tbsv {
                triangle,
                transform,
                diagonal,
                a,
                x,
            } => {
                let a = Operand::Band {
                    matrix: a,
                    triangle: Some(triangle),
                };

                triangular_vector!(transform, diagonal, a, x);
            }
            BlasCall::Tpsv {
                transform,
                diagonal,
                a,
                x,
            } => {
                let a = Operand::Packed(a);

                triangular_vector!(transform, diagonal, a, x);
            }
            BlasCall::Geru { alpha, x, y, a } => {
                let operation = BlasOperation::Geru;
                let a = OperandMut::Dense(a);
                let triangle = match &a {
                    OperandMut::Packed(a) => a.triangle(),
                    OperandMut::Dense(_) => Triangle::Lower,
                };
                let y = Some(y);

                rank_update!(operation, triangle, alpha, x, y, a);
            }
            BlasCall::Gerc { alpha, x, y, a } => {
                let operation = BlasOperation::Gerc;
                let a = OperandMut::Dense(a);
                let triangle = match &a {
                    OperandMut::Packed(a) => a.triangle(),
                    OperandMut::Dense(_) => Triangle::Lower,
                };
                let y = Some(y);

                rank_update!(operation, triangle, alpha, x, y, a);
            }
            BlasCall::Syr {
                triangle,
                alpha,
                x,
                a,
            } => {
                let operation = BlasOperation::Syr;
                let a = OperandMut::Dense(a);
                let y: Option<BlasVector<'_, S>> = None;

                rank_update!(operation, triangle, alpha, x, y, a);
            }
            BlasCall::Syr2 {
                triangle,
                alpha,
                x,
                y,
                a,
            } => {
                let operation = BlasOperation::Syr2;
                let a = OperandMut::Dense(a);
                let y = Some(y);

                rank_update!(operation, triangle, alpha, x, y, a);
            }
            BlasCall::Her {
                triangle,
                alpha,
                x,
                a,
            } => {
                let operation = BlasOperation::Her;
                let alpha = S::from_real(alpha);
                let a = OperandMut::Dense(a);
                let y: Option<BlasVector<'_, S>> = None;

                rank_update!(operation, triangle, alpha, x, y, a);
            }
            BlasCall::Her2 {
                triangle,
                alpha,
                x,
                y,
                a,
            } => {
                let operation = BlasOperation::Her2;
                let a = OperandMut::Dense(a);
                let y = Some(y);

                rank_update!(operation, triangle, alpha, x, y, a);
            }
            BlasCall::Spr { alpha, x, a } => {
                let operation = BlasOperation::Spr;
                let a = OperandMut::Packed(a);
                let triangle = match &a {
                    OperandMut::Packed(a) => a.triangle(),
                    OperandMut::Dense(_) => Triangle::Lower,
                };
                let y: Option<BlasVector<'_, S>> = None;

                rank_update!(operation, triangle, alpha, x, y, a);
            }
            BlasCall::Spr2 { alpha, x, y, a } => {
                let operation = BlasOperation::Spr2;
                let a = OperandMut::Packed(a);
                let triangle = match &a {
                    OperandMut::Packed(a) => a.triangle(),
                    OperandMut::Dense(_) => Triangle::Lower,
                };
                let y = Some(y);

                rank_update!(operation, triangle, alpha, x, y, a);
            }
            BlasCall::Hpr { alpha, x, a } => {
                let operation = BlasOperation::Hpr;
                let alpha = S::from_real(alpha);
                let a = OperandMut::Packed(a);
                let triangle = match &a {
                    OperandMut::Packed(a) => a.triangle(),
                    OperandMut::Dense(_) => Triangle::Lower,
                };
                let y: Option<BlasVector<'_, S>> = None;

                rank_update!(operation, triangle, alpha, x, y, a);
            }
            BlasCall::Hpr2 { alpha, x, y, a } => {
                let operation = BlasOperation::Hpr2;
                let a = OperandMut::Packed(a);
                let triangle = match &a {
                    OperandMut::Packed(a) => a.triangle(),
                    OperandMut::Dense(_) => Triangle::Lower,
                };
                let y = Some(y);

                rank_update!(operation, triangle, alpha, x, y, a);
            }
            BlasCall::Symm {
                side,
                triangle,
                alpha,
                a,
                b,
                beta,
                mut c,
            } => {
                let operation = BlasOperation::Symm;

                let raw = |row, col| {
                    let v = if triangle_contains(triangle, row, col) {
                        matrix_get(a, row, col)
                    } else {
                        let v = matrix_get(a, col, row);
                        if operation == BlasOperation::Hemm {
                            v.conj()
                        } else {
                            v
                        }
                    };
                    if operation == BlasOperation::Hemm && row == col {
                        S::from_real(v.real_component())
                    } else {
                        v
                    }
                };
                let order = a.rows();
                for row in 0..c.rows() {
                    for col in 0..c.cols() {
                        let sum = (0..order).fold(S::ZERO, |sum, k| {
                            sum + if side == scioxide_linalg::MatrixSide::Left {
                                raw(row, k) * matrix_get(b, k, col)
                            } else {
                                matrix_get(b, row, k) * raw(k, col)
                            }
                        });
                        let prior = if beta == S::ZERO {
                            S::ZERO
                        } else {
                            beta * matrix_mut_get(&c, row, col)
                        };
                        matrix_set(&mut c, row, col, alpha * sum + prior);
                    }
                }
            }
            BlasCall::Hemm {
                side,
                triangle,
                alpha,
                a,
                b,
                beta,
                mut c,
            } => {
                let operation = BlasOperation::Hemm;

                let raw = |row, col| {
                    let v = if triangle_contains(triangle, row, col) {
                        matrix_get(a, row, col)
                    } else {
                        let v = matrix_get(a, col, row);
                        if operation == BlasOperation::Hemm {
                            v.conj()
                        } else {
                            v
                        }
                    };
                    if operation == BlasOperation::Hemm && row == col {
                        S::from_real(v.real_component())
                    } else {
                        v
                    }
                };
                let order = a.rows();
                for row in 0..c.rows() {
                    for col in 0..c.cols() {
                        let sum = (0..order).fold(S::ZERO, |sum, k| {
                            sum + if side == scioxide_linalg::MatrixSide::Left {
                                raw(row, k) * matrix_get(b, k, col)
                            } else {
                                matrix_get(b, row, k) * raw(k, col)
                            }
                        });
                        let prior = if beta == S::ZERO {
                            S::ZERO
                        } else {
                            beta * matrix_mut_get(&c, row, col)
                        };
                        matrix_set(&mut c, row, col, alpha * sum + prior);
                    }
                }
            }
            BlasCall::Syr2k {
                triangle,
                transform,
                alpha,
                a,
                b,
                beta,
                mut c,
            } => {
                let operation = BlasOperation::Syr2k;

                let hermitian = operation == BlasOperation::Her2k;
                let (order, shared) = matrix_logical_shape(a, transform);
                for row in 0..order {
                    for col in 0..order {
                        if !triangle_contains(triangle, row, col) {
                            continue;
                        }
                        let sum = (0..shared).fold(S::ZERO, |sum, k| {
                            let ar = op_get(a, transform, row, k);
                            let ac = op_get(a, transform, col, k);
                            let br = op_get(b, transform, row, k);
                            let bc = op_get(b, transform, col, k);
                            sum + if hermitian {
                                alpha * ar * bc.conj() + alpha.conj() * br * ac.conj()
                            } else {
                                alpha * (ar * bc + br * ac)
                            }
                        });
                        let prior = if beta == S::ZERO {
                            S::ZERO
                        } else {
                            beta * matrix_mut_get(&c, row, col)
                        };
                        let v = sum + prior;
                        matrix_set(
                            &mut c,
                            row,
                            col,
                            if hermitian && row == col {
                                S::from_real(v.real_component())
                            } else {
                                v
                            },
                        );
                    }
                }
            }
            BlasCall::Her2k {
                triangle,
                transform,
                alpha,
                a,
                b,
                beta,
                mut c,
            } => {
                let operation = BlasOperation::Her2k;
                let beta = S::from_real(beta);

                let hermitian = operation == BlasOperation::Her2k;
                let (order, shared) = matrix_logical_shape(a, transform);
                for row in 0..order {
                    for col in 0..order {
                        if !triangle_contains(triangle, row, col) {
                            continue;
                        }
                        let sum = (0..shared).fold(S::ZERO, |sum, k| {
                            let ar = op_get(a, transform, row, k);
                            let ac = op_get(a, transform, col, k);
                            let br = op_get(b, transform, row, k);
                            let bc = op_get(b, transform, col, k);
                            sum + if hermitian {
                                alpha * ar * bc.conj() + alpha.conj() * br * ac.conj()
                            } else {
                                alpha * (ar * bc + br * ac)
                            }
                        });
                        let prior = if beta == S::ZERO {
                            S::ZERO
                        } else {
                            beta * matrix_mut_get(&c, row, col)
                        };
                        let v = sum + prior;
                        matrix_set(
                            &mut c,
                            row,
                            col,
                            if hermitian && row == col {
                                S::from_real(v.real_component())
                            } else {
                                v
                            },
                        );
                    }
                }
            }
        }
        Ok(())
    }
}

fn call_operation<S: NumericScalar>(call: &BlasCall<'_, S>) -> BlasOperation {
    call.operation()
}
fn operation_bit<S: NumericScalar>(call: &BlasCall<'_, S>) -> u64 {
    1 << call_operation(call) as u8
}

#[test]
fn injected_level_one_absolute_reductions_match_reference() {
    let provider = CblasProvider::new();
    let values = [Complex64::new(1.0, -2.0), Complex64::new(-4.0, 1.0)];
    let vector = VectorView::from_slice(&values);
    assert_eq!(
        vector
            .asum_scoped(
                &scioxide_linalg::LinalgExecutionOptions::new(),
                provider.context()
            )
            .unwrap(),
        8.0
    );
    assert_eq!(
        vector
            .iamax_scoped(
                &scioxide_linalg::LinalgExecutionOptions::new(),
                provider.context()
            )
            .unwrap(),
        Some(1)
    );
    assert_ne!(
        provider.operations.load(Ordering::Relaxed) & (1 << BlasOperation::Asum as u8),
        0
    );
    assert_ne!(
        provider.operations.load(Ordering::Relaxed) & (1 << BlasOperation::Iamax as u8),
        0
    );
}

#[test]
fn injected_band_transpose_matches_reference() {
    let provider = CblasProvider::new();
    let data = [0.0_f64, 1.0, 3.0, 2.0, 4.0, 0.0, 5.0, 0.0, 0.0];
    let band = BandedMatrixView::new(2, 3, 1, 1, 3, &data).unwrap();
    let x = [7.0, 11.0];
    let mut y = [0.0; 3];
    let mut scratch = [0.0; 3];
    band.mul_vec_in_place(
        StructuredOperation::Transpose,
        1.0,
        VectorView::from_slice(&x),
        0.0,
        &scioxide_linalg::LinalgExecutionOptions::new(),
        VectorViewMut::from_slice(&mut y),
        &mut scioxide_linalg::LinalgWorkspace::from_scalar_slice(&mut scratch),
        provider.context(),
    )
    .unwrap();
    assert_eq!(y, [40.0, 58.0, 55.0]);
    assert_ne!(
        provider.operations.load(Ordering::Relaxed) & (1 << BlasOperation::Gbmv as u8),
        0
    );
}

#[test]
fn extended_provider_failure_and_decline_preserve_band_output_contract() {
    let provider = CblasProvider::new();
    provider.fail_after_write.store(true, Ordering::Relaxed);
    let data = [0.0_f64, 2.0, 1.0, 3.0];
    let band = BandedMatrixView::new(2, 2, 0, 1, 2, &data).unwrap();
    let input = [4.0, 5.0];
    let mut output = [71.0, 72.0];
    let mut scratch = [0.0; 2];
    assert!(
        band.mul_vec_in_place(
            StructuredOperation::Normal,
            1.0,
            VectorView::from_slice(&input),
            0.0,
            &scioxide_linalg::LinalgExecutionOptions::new(),
            VectorViewMut::from_slice(&mut output),
            &mut scioxide_linalg::LinalgWorkspace::from_scalar_slice(&mut scratch),
            provider.context()
        )
        .is_err()
    );
    assert_eq!(output, [71.0, 72.0]);

    let unsupported =
        CblasProvider::with_capabilities(BlasCapabilities::empty(BlasIntegerWidth::Lp64));
    let fallback = LinalgExecutionContext::sequential().with_blas(BlasDispatch::new(
        &unsupported,
        BlasFallbackPolicy::Reference,
    ));
    band.mul_vec_in_place(
        StructuredOperation::Normal,
        1.0,
        VectorView::from_slice(&input),
        0.0,
        &scioxide_linalg::LinalgExecutionOptions::new(),
        VectorViewMut::from_slice(&mut output),
        &mut scioxide_linalg::LinalgWorkspace::from_scalar_slice(&mut scratch),
        fallback,
    )
    .unwrap();
    assert_eq!(output, [13.0, 15.0]);
}

#[test]
fn injected_dense_and_packed_rank_updates_commit_once() {
    let provider = CblasProvider::new();
    let mut dense = [1.0_f64, 2.0, 3.0, 4.0];
    let mut scratch = [0.0; 4];
    scioxide_linalg::MatrixViewMut::rank_one_update_in_place(
        0.5,
        VectorView::from_slice(&[2.0, -1.0]),
        VectorView::from_slice(&[3.0, 4.0]),
        &scioxide_linalg::LinalgExecutionOptions::new(),
        scioxide_linalg::MatrixViewMut::from_slice(2, 2, &mut dense).unwrap(),
        &mut scioxide_linalg::LinalgWorkspace::from_scalar_slice(&mut scratch),
        provider.context(),
    )
    .unwrap();
    assert_eq!(dense, [4.0, 6.0, 1.5, 2.0]);

    let mut packed_data = [
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(2.0, 0.0),
    ];
    let mut packed = scioxide_linalg::PackedMatrixViewMut::new(
        2,
        scioxide_core::Triangle::Lower,
        scioxide_linalg::PackedKind::Hermitian,
        &mut packed_data,
    )
    .unwrap();
    let x = [Complex64::new(1.0, 1.0), Complex64::new(2.0, -1.0)];
    let mut scratch = [Complex64::new(0.0, 0.0); 3];
    scioxide_linalg::PackedMatrixViewMut::hermitian_rank_one_update_in_place(
        Complex64::new(0.5, 0.0),
        VectorView::from_slice(&x),
        &scioxide_linalg::LinalgExecutionOptions::new(),
        packed.reborrow(),
        &mut scioxide_linalg::LinalgWorkspace::from_scalar_slice(&mut scratch),
        provider.context(),
    )
    .unwrap();
    assert_eq!(packed.get(1, 0).unwrap(), Complex64::new(0.5, -1.5));
    assert_ne!(
        provider.operations.load(Ordering::Relaxed) & (1 << BlasOperation::Geru as u8),
        0
    );
    assert_ne!(
        provider.operations.load(Ordering::Relaxed) & (1 << BlasOperation::Hpr as u8),
        0
    );
}

#[test]
fn injected_symmetric_product_and_hermitian_rank_two_k_commit() {
    let provider = CblasProvider::new();
    let a_data = [2.0_f64, f64::NAN, 1.0, 3.0];
    let b_data = [1.0, 2.0, 3.0, 4.0];
    let a = MatrixView::from_slice(2, 2, &a_data).unwrap();
    let b = MatrixView::from_slice(2, 2, &b_data).unwrap();
    let mut out = [0.0; 4];
    let mut scratch = LinalgWorkspace::new();
    a.symmetric_mul_mat_in_place(
        b,
        scioxide_linalg::MatrixSide::Left,
        scioxide_core::Triangle::Lower,
        1.0,
        0.0,
        &scioxide_linalg::LinalgExecutionOptions::new(),
        scioxide_linalg::MatrixViewMut::from_slice(2, 2, &mut out).unwrap(),
        &mut scratch,
        provider.context(),
    )
    .unwrap();
    assert_eq!(out, [5.0, 8.0, 10.0, 14.0]);

    let a_data = [Complex64::new(1.0, 1.0), Complex64::new(2.0, 0.0)];
    let b_data = [Complex64::new(3.0, 0.0), Complex64::new(0.0, 1.0)];
    let a = MatrixView::from_slice(2, 1, &a_data).unwrap();
    let b = MatrixView::from_slice(2, 1, &b_data).unwrap();
    let mut out = [Complex64::new(0.0, 0.0); 4];
    let mut scratch = LinalgWorkspace::new();
    scioxide_linalg::MatrixViewMut::hermitian_rank_two_k_in_place(
        a,
        b,
        scioxide_core::Triangle::Lower,
        StructuredOperation::Normal,
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        &scioxide_linalg::LinalgExecutionOptions::new(),
        scioxide_linalg::MatrixViewMut::from_slice(2, 2, &mut out).unwrap(),
        &mut scratch,
        provider.context(),
    )
    .unwrap();
    assert_eq!(out[2], Complex64::new(7.0, 1.0));
    assert_ne!(
        provider.operations.load(Ordering::Relaxed) & (1 << BlasOperation::Symm as u8),
        0
    );
    assert_ne!(
        provider.operations.load(Ordering::Relaxed) & (1 << BlasOperation::Her2k as u8),
        0
    );
}

#[test]
fn injected_packed_triangular_vector_solve_matches_reference() {
    let provider = CblasProvider::new();
    let data = [
        Complex64::new(f64::NAN, 0.0),
        Complex64::new(1.0, 2.0),
        Complex64::new(f64::NAN, 0.0),
    ];
    let packed = scioxide_linalg::PackedMatrixView::new(
        2,
        scioxide_core::Triangle::Lower,
        scioxide_linalg::PackedKind::Triangular,
        &data,
    )
    .unwrap();
    let plan =
        scioxide_linalg::TriangularPlan::from_packed(packed, scioxide_core::DiagonalKind::Unit)
            .unwrap();
    let input = [Complex64::new(2.0, 0.0), Complex64::new(3.0, 0.0)];
    // `Lᴴ input` for the unit lower triangle with subdiagonal `1 + 2i`.
    let product = [Complex64::new(5.0, -6.0), Complex64::new(3.0, 0.0)];
    let mut scratch = [Complex64::new(0.0, 0.0); 2];
    let mut solved = [Complex64::new(0.0, 0.0); 2];
    let report = plan
        .solve_into(
            StructuredOperation::Adjoint,
            VectorView::from_slice(&product),
            &scioxide_linalg::DirectSolveOptions::new(scioxide_linalg::RankPolicy::Exact),
            VectorViewMut::from_slice(&mut solved),
            &mut LinalgWorkspace::from_scalar_slice(&mut scratch),
            provider.context(),
        )
        .unwrap();
    assert_eq!(
        report.termination(),
        scioxide_linalg::DirectSolveTermination::Accepted
    );
    assert_eq!(solved, input);
    assert_ne!(
        provider.operations.load(Ordering::Relaxed) & (1 << BlasOperation::Tpsv as u8),
        0
    );
}

#[test]
fn public_triangular_matmul_uses_existing_provider_call() {
    let provider = CblasProvider::new();
    let factor_data = [2.0_f64, 1.0, f64::NAN, 3.0];
    let factor = MatrixView::from_slice(2, 2, &factor_data).unwrap();
    let mut data = [1.0, 2.0, 3.0, 4.0];
    let mut scratch = LinalgWorkspace::new();
    factor
        .triangular_mul_mat_in_place(
            scioxide_linalg::MatrixSide::Right,
            scioxide_core::Triangle::Upper,
            scioxide_core::DiagonalKind::Explicit,
            StructuredOperation::Normal,
            1.0,
            &scioxide_linalg::LinalgExecutionOptions::new(),
            scioxide_linalg::MatrixViewMut::from_slice(2, 2, &mut data).unwrap(),
            &mut scratch,
            provider.context(),
        )
        .unwrap();
    assert_eq!(data, [2.0, 7.0, 6.0, 15.0]);
    assert_ne!(
        provider.operations.load(Ordering::Relaxed) & (1 << BlasOperation::Trmm as u8),
        0
    );
}

#[test]
fn exact_action_reductions_never_reach_an_inexact_provider() {
    use scioxide_core::Accuracy;
    use scioxide_linalg::{
        OperatorStructure, PartialSpectrumBuffer, PartialSpectrumSolver, PartialSvdBuffer,
        PartialSvdSolver,
    };
    let matrix =
        Matrix::from_vec(3, 3, vec![2.0_f64, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 4.0]).unwrap();
    let operator = matrix.view().try_finite().unwrap();
    let initial = [1.0, 1.0, 1.0];
    for accuracy in [Accuracy::Standard, Accuracy::Exact] {
        // The provider declares four-ULP reductions, which Exact action norms and
        // dots must never use; the reference fallback keeps the native result.
        let provider = CblasProvider::new();
        let context = || {
            LinalgExecutionContext::sequential()
                .with_blas(BlasDispatch::new(&provider, BlasFallbackPolicy::Reference))
        };
        let execution = LinalgExecutionOptions::new().with_accuracy(accuracy);
        let spectrum = PartialSpectrumSolver::new(1)
            .with_structure(OperatorStructure::Hermitian)
            .with_execution_options(execution);
        let mut workspace = spectrum.workspace(&operator).unwrap();
        let mut pairs = PartialSpectrumBuffer::new(3, 1).unwrap();
        spectrum
            .solve_into(
                &operator,
                &initial[..],
                &mut pairs,
                &mut workspace,
                context(),
            )
            .unwrap();
        let svd = PartialSvdSolver::new(1).with_execution_options(execution);
        let mut workspace = svd.workspace(&operator).unwrap();
        let mut triplets = PartialSvdBuffer::new(3, 3, 1).unwrap();
        svd.solve_into(
            &operator,
            &initial[..],
            &mut triplets,
            &mut workspace,
            context(),
        )
        .unwrap();
        // Documented Standard-only dense stages may still use the provider.
        let reductions = [
            BlasOperation::Dotu,
            BlasOperation::Dotc,
            BlasOperation::Norm2,
        ]
        .into_iter()
        .fold(0, |mask, operation| mask | 1 << operation as u8);
        let used = provider.operations.load(Ordering::Relaxed) & reductions;
        if accuracy == Accuracy::Exact {
            assert_eq!(used, 0);
        } else {
            assert_ne!(used, 0);
        }
    }
}

#[test]
fn masked_gemm_validates_only_the_published_triangle() {
    // Blocked Cholesky updates its trailing block with a lower-masked GEMM, so
    // a provider that leaves NaN above the diagonal must not fail the factor.
    let n = 130;
    let a = Matrix::from_vec(
        n,
        n,
        (0..n * n)
            .map(|index| {
                let (row, col) = (index / n, index % n);
                if row == col {
                    n as f64
                } else {
                    1.0 / (1.0 + row.abs_diff(col) as f64)
                }
            })
            .collect(),
    )
    .unwrap();
    let options = LinalgExecutionOptions::new();
    let reference = CholeskyDecomposition::factor(
        &a,
        scioxide_linalg::HermitianInput::Exact,
        scioxide_linalg::CholeskyPivoting::None,
    )
    .unwrap();
    let provider = CblasProvider::new();
    provider.poison_upper_gemm.store(true, Ordering::Relaxed);
    let mut factor = CholeskyDecomposition::empty();
    CholeskyDecomposition::factor_into(
        &a,
        scioxide_linalg::HermitianInput::Exact,
        scioxide_linalg::CholeskyPivoting::None,
        &options,
        &mut factor,
        &mut LinalgWorkspace::new(),
        provider.context(),
    )
    .unwrap();
    assert_ne!(
        provider.operations.load(Ordering::Relaxed) & (1 << BlasOperation::Gemm as u8),
        0
    );
    for (&actual, &expected) in factor
        .l_matrix()
        .as_slice()
        .iter()
        .zip(reference.l_matrix().as_slice())
    {
        assert!((actual - expected).abs() <= 1e-12 * expected.abs().max(1.0));
    }
}

#[test]
fn hermitian_rank_k_provider_validates_only_selected_triangle_and_real_diagonal() {
    let provider = CblasProvider::new();
    let a_data = [Complex64::new(1.0, 1.0), Complex64::new(2.0, 0.0)];
    let a = MatrixView::from_slice(2, 1, &a_data).unwrap();
    let unused = Complex64::new(f64::NAN, 0.0);
    let mut output = [
        Complex64::new(0.0, 0.0),
        unused,
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
    ];
    let mut workspace = LinalgWorkspace::new();
    scioxide_linalg::MatrixViewMut::hermitian_rank_k_in_place(
        a,
        scioxide_core::Triangle::Lower,
        StructuredOperation::Normal,
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        &scioxide_linalg::LinalgExecutionOptions::new()
            .with_execution_policy(ExecutionPolicy::Sequential),
        scioxide_linalg::MatrixViewMut::from_slice(2, 2, &mut output).unwrap(),
        &mut workspace,
        provider.context(),
    )
    .unwrap();
    assert_eq!(output[0], Complex64::new(2.0, 0.0));
    assert!(output[1].re.is_nan());

    let prior = [output[0], output[2], output[3]];
    provider.malformed_result.store(true, Ordering::Relaxed);
    assert!(
        scioxide_linalg::MatrixViewMut::hermitian_rank_k_in_place(
            a,
            scioxide_core::Triangle::Lower,
            StructuredOperation::Normal,
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            &scioxide_linalg::LinalgExecutionOptions::new()
                .with_execution_policy(ExecutionPolicy::Sequential),
            scioxide_linalg::MatrixViewMut::from_slice(2, 2, &mut output).unwrap(),
            &mut workspace,
            provider.context()
        )
        .is_err()
    );
    assert_eq!([output[0], output[2], output[3]], prior);
    assert!(output[1].re.is_nan());
}

#[test]
fn extreme_modified_rotation_bypasses_provider_and_preserves_norm() {
    let provider = CblasProvider::new();
    let result = scioxide_linalg::modified_givens_rotation_scoped(
        1e-300_f64,
        1e-300,
        1e-100,
        1e-100,
        &scioxide_linalg::LinalgExecutionOptions::new(),
        LinalgExecutionContext::sequential()
            .with_blas(BlasDispatch::new(&provider, BlasFallbackPolicy::Reference)),
    )
    .unwrap();
    assert_eq!(provider.calls.load(Ordering::Relaxed), 0);
    let error = scioxide_linalg::modified_givens_rotation_scoped(
        1e-300_f64,
        1e-300,
        1e-100,
        1e-100,
        &scioxide_linalg::LinalgExecutionOptions::new(),
        provider.context(),
    )
    .unwrap_err();
    assert_eq!(
        error.kind(),
        scioxide_core::SciOxideErrorKind::CapabilityUnavailable
    );
    assert_eq!(provider.calls.load(Ordering::Relaxed), 0);
    let expected = 2.0_f64.sqrt() * 1e-250;
    assert!((result.d1().sqrt() * result.x1() / expected - 1.0).abs() < 1e-14);
}

#[test]
fn injected_level_one_rotations_and_swap_use_typed_calls() {
    let provider = CblasProvider::new();
    let generated = scioxide_linalg::givens_rotation_scoped(
        3.0_f64,
        4.0,
        &scioxide_linalg::LinalgExecutionOptions::new(),
        provider.context(),
    )
    .unwrap();
    let rotation = generated.rotation();
    assert_eq!(generated.r(), 5.0);
    let generated = scioxide_linalg::modified_givens_rotation_scoped(
        1.0_f64,
        1.0,
        3.0,
        4.0,
        &scioxide_linalg::LinalgExecutionOptions::new(),
        provider.context(),
    )
    .unwrap();
    assert_eq!(generated.rotation().coefficients(), [0.75, 1.0, -1.0, 0.75]);

    let mut x = [3.0_f64, 6.0];
    let mut y = [4.0_f64, 8.0];
    let mut workspace = scioxide_linalg::LinalgWorkspace::new();
    scioxide_linalg::VectorViewMut::rotate_in_place(
        rotation,
        &scioxide_linalg::LinalgExecutionOptions::new(),
        VectorViewMut::from_slice(&mut x),
        VectorViewMut::from_slice(&mut y),
        &mut workspace,
        provider.context(),
    )
    .unwrap();
    assert!(
        x.iter()
            .zip([5.0, 10.0])
            .all(|(actual, expected)| (*actual - expected).abs() < 1e-13)
    );
    assert!(y.iter().all(|value| value.abs() < 1e-13));

    scioxide_linalg::VectorViewMut::modified_rotate_in_place(
        generated.rotation(),
        &scioxide_linalg::LinalgExecutionOptions::new(),
        VectorViewMut::from_slice(&mut x),
        VectorViewMut::from_slice(&mut y),
        &mut workspace,
        provider.context(),
    )
    .unwrap();
    assert!(
        x.iter()
            .zip([3.75, 7.5])
            .all(|(actual, expected)| (*actual - expected).abs() < 1e-13)
    );
    assert!(
        y.iter()
            .zip([-5.0, -10.0])
            .all(|(actual, expected)| (*actual - expected).abs() < 1e-13)
    );
    scioxide_linalg::VectorViewMut::swap_in_place(
        &scioxide_linalg::LinalgExecutionOptions::new(),
        VectorViewMut::from_slice(&mut x),
        VectorViewMut::from_slice(&mut y),
        &mut workspace,
        provider.context(),
    )
    .unwrap();
    assert!(
        x.iter()
            .zip([-5.0, -10.0])
            .all(|(actual, expected)| (*actual - expected).abs() < 1e-13)
    );
    assert!(
        y.iter()
            .zip([3.75, 7.5])
            .all(|(actual, expected)| (*actual - expected).abs() < 1e-13)
    );
    for bit in [
        BlasOperation::Swap,
        BlasOperation::Rot,
        BlasOperation::Rotm,
        BlasOperation::Rotg,
        BlasOperation::Rotmg,
    ] {
        assert_ne!(
            provider.operations.load(Ordering::Relaxed) & (1 << bit as u8),
            0
        );
    }
}

#[test]
fn injected_widened_and_offset_binary32_dots_are_typed() {
    let provider = CblasProvider::new();
    let left = VectorView::from_slice(&[1.0_f32, 2.0, 3.0]);
    let right = VectorView::from_slice(&[4.0_f32, 5.0, 6.0]);
    assert_eq!(
        left.dot_wide_scoped(
            right,
            &scioxide_linalg::LinalgExecutionOptions::new(),
            provider.context()
        )
        .unwrap(),
        32.0
    );
    assert_eq!(
        left.dot_plus_scoped(
            right,
            2.0,
            &scioxide_linalg::LinalgExecutionOptions::new(),
            provider.context()
        )
        .unwrap(),
        34.0
    );
    assert_ne!(
        provider.operations.load(Ordering::Relaxed) & (1 << BlasOperation::WideDot as u8),
        0
    );
    assert_ne!(
        provider.operations.load(Ordering::Relaxed) & (1 << BlasOperation::OffsetDot as u8),
        0
    );
}

impl BlasProvider for CblasProvider {
    fn name(&self) -> &'static str {
        "rust-cblas"
    }

    fn capabilities(&self) -> BlasCapabilities {
        self.capabilities.clone()
    }

    fn execute_f32(
        &self,
        call: BlasCall<'_, f32>,
        workers: NonZeroUsize,
    ) -> Result<(), CapabilityError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.workers.store(workers.get(), Ordering::Relaxed);
        self.execute(call)
    }

    fn execute_f64(
        &self,
        call: BlasCall<'_, f64>,
        workers: NonZeroUsize,
    ) -> Result<(), CapabilityError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.workers.store(workers.get(), Ordering::Relaxed);
        if self.malformed_result.load(Ordering::Relaxed) {
            let conjugate_left = call.operation() == BlasOperation::Dotc;
            match call {
                BlasCall::Dotu { result, .. } | BlasCall::Dotc { result, .. } => {
                    let operation = if conjugate_left {
                        1 << BlasOperation::Dotc as u8
                    } else {
                        1 << BlasOperation::Dotu as u8
                    };
                    self.operations.fetch_or(operation, Ordering::Relaxed);
                    *result = f64::NAN;
                    Ok(())
                }
                call => self.execute(call),
            }
        } else {
            self.execute(call)
        }
    }

    fn execute_complex32(
        &self,
        call: BlasCall<'_, Complex32>,
        workers: NonZeroUsize,
    ) -> Result<(), CapabilityError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.workers.store(workers.get(), Ordering::Relaxed);
        self.execute(call)
    }

    fn execute_complex64(
        &self,
        call: BlasCall<'_, Complex64>,
        workers: NonZeroUsize,
    ) -> Result<(), CapabilityError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.workers.store(workers.get(), Ordering::Relaxed);
        self.execute(call)
    }
}

fn vector_get<S: NumericScalar>(vector: BlasVector<'_, S>, index: usize) -> S {
    let value = vector.values()[index * vector.stride()];
    if vector.is_conjugated() {
        value.conj()
    } else {
        value
    }
}

fn vector_mut_get<S: NumericScalar>(vector: &BlasVectorMut<'_, S>, index: usize) -> S {
    vector.values()[index * vector.stride()]
}

fn vector_set<S: NumericScalar>(vector: &mut BlasVectorMut<'_, S>, index: usize, value: S) {
    let stride = vector.stride();
    vector.values_mut()[index * stride] = value;
}

fn matrix_logical_shape<S: NumericScalar>(
    matrix: BlasMatrix<'_, S>,
    transform: BlasTransform,
) -> (usize, usize) {
    match transform {
        BlasTransform::None | BlasTransform::Conjugate => (matrix.rows(), matrix.cols()),
        BlasTransform::Transpose | BlasTransform::Adjoint => (matrix.cols(), matrix.rows()),
    }
}

fn op_get<S: NumericScalar>(
    matrix: BlasMatrix<'_, S>,
    transform: BlasTransform,
    row: usize,
    col: usize,
) -> S {
    let (stored_row, stored_col, conjugate) = match transform {
        BlasTransform::None => (row, col, false),
        BlasTransform::Transpose => (col, row, false),
        BlasTransform::Conjugate => (row, col, true),
        BlasTransform::Adjoint => (col, row, true),
    };
    let index = match matrix.layout() {
        BlasLayout::RowMajor => stored_row * matrix.leading_dimension() + stored_col,
        BlasLayout::ColumnMajor => stored_col * matrix.leading_dimension() + stored_row,
    };
    let value = matrix.values()[index];
    if conjugate { value.conj() } else { value }
}

fn matrix_mut_get<S: NumericScalar>(matrix: &BlasMatrixMut<'_, S>, row: usize, col: usize) -> S {
    let index = match matrix.layout() {
        BlasLayout::RowMajor => row * matrix.leading_dimension() + col,
        BlasLayout::ColumnMajor => col * matrix.leading_dimension() + row,
    };
    matrix.values()[index]
}

fn matrix_set<S: NumericScalar>(
    matrix: &mut BlasMatrixMut<'_, S>,
    row: usize,
    col: usize,
    value: S,
) {
    let index = match matrix.layout() {
        BlasLayout::RowMajor => row * matrix.leading_dimension() + col,
        BlasLayout::ColumnMajor => col * matrix.leading_dimension() + row,
    };
    matrix.values_mut()[index] = value;
}

fn triangle_contains(triangle: scioxide_core::Triangle, row: usize, col: usize) -> bool {
    match triangle {
        scioxide_core::Triangle::Lower => row >= col,
        scioxide_core::Triangle::Upper => row <= col,
    }
}

fn matrix_get<S: NumericScalar>(matrix: BlasMatrix<'_, S>, row: usize, col: usize) -> S {
    op_get(matrix, BlasTransform::None, row, col)
}

fn triangular_get<S: NumericScalar>(
    matrix: BlasMatrix<'_, S>,
    triangle: scioxide_core::Triangle,
    diagonal: scioxide_core::DiagonalKind,
    transform: BlasTransform,
    row: usize,
    col: usize,
) -> S {
    let (r, c) = if matches!(transform, BlasTransform::Transpose | BlasTransform::Adjoint) {
        (col, row)
    } else {
        (row, col)
    };
    let value = if r == c && diagonal == scioxide_core::DiagonalKind::Unit {
        S::ONE
    } else if triangle_contains(triangle, r, c) {
        matrix_get(matrix, r, c)
    } else {
        S::ZERO
    };
    if matches!(transform, BlasTransform::Adjoint | BlasTransform::Conjugate) {
        value.conj()
    } else {
        value
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_triangular<S: NumericScalar>(
    solve: bool,
    side: scioxide_linalg::MatrixSide,
    triangle: scioxide_core::Triangle,
    diagonal: scioxide_core::DiagonalKind,
    transform: BlasTransform,
    alpha: S,
    a: BlasMatrix<'_, S>,
    b: &mut BlasMatrixMut<'_, S>,
) {
    let rows = b.rows();
    let cols = b.cols();
    let mut prior = Vec::with_capacity(rows * cols);
    for row in 0..rows {
        for col in 0..cols {
            prior.push(matrix_mut_get(b, row, col));
        }
    }
    (0..rows).for_each(|row| {
        (0..cols).for_each(|col| matrix_set(b, row, col, alpha * prior[row * cols + col]));
    });
    if !solve {
        (0..rows).for_each(|row| {
            (0..cols).for_each(|col| {
                let product = match side {
                    scioxide_linalg::MatrixSide::Left => (0..rows).fold(S::ZERO, |sum, inner| {
                        sum + triangular_get(a, triangle, diagonal, transform, row, inner)
                            * prior[inner * cols + col]
                    }),
                    scioxide_linalg::MatrixSide::Right => (0..cols).fold(S::ZERO, |sum, inner| {
                        sum + prior[row * cols + inner]
                            * triangular_get(a, triangle, diagonal, transform, inner, col)
                    }),
                };
                matrix_set(b, row, col, alpha * product);
            });
        });
        return;
    }

    let effective_triangle =
        if matches!(transform, BlasTransform::Transpose | BlasTransform::Adjoint) {
            flip_triangle(triangle)
        } else {
            triangle
        };
    match (side, effective_triangle) {
        (scioxide_linalg::MatrixSide::Left, scioxide_core::Triangle::Lower) => {
            for row in 0..rows {
                for col in 0..cols {
                    let value = (0..row).fold(matrix_mut_get(b, row, col), |value, inner| {
                        value
                            - triangular_get(a, triangle, diagonal, transform, row, inner)
                                * matrix_mut_get(b, inner, col)
                    });
                    matrix_set(
                        b,
                        row,
                        col,
                        value / triangular_get(a, triangle, diagonal, transform, row, row),
                    );
                }
            }
        }
        (scioxide_linalg::MatrixSide::Left, scioxide_core::Triangle::Upper) => {
            for row in (0..rows).rev() {
                for col in 0..cols {
                    let value =
                        (row + 1..rows).fold(matrix_mut_get(b, row, col), |value, inner| {
                            value
                                - triangular_get(a, triangle, diagonal, transform, row, inner)
                                    * matrix_mut_get(b, inner, col)
                        });
                    matrix_set(
                        b,
                        row,
                        col,
                        value / triangular_get(a, triangle, diagonal, transform, row, row),
                    );
                }
            }
        }
        (scioxide_linalg::MatrixSide::Right, scioxide_core::Triangle::Lower) => {
            for col in (0..cols).rev() {
                for row in 0..rows {
                    let value =
                        (col + 1..cols).fold(matrix_mut_get(b, row, col), |value, inner| {
                            value
                                - matrix_mut_get(b, row, inner)
                                    * triangular_get(a, triangle, diagonal, transform, inner, col)
                        });
                    matrix_set(
                        b,
                        row,
                        col,
                        value / triangular_get(a, triangle, diagonal, transform, col, col),
                    );
                }
            }
        }
        (scioxide_linalg::MatrixSide::Right, scioxide_core::Triangle::Upper) => {
            for col in 0..cols {
                for row in 0..rows {
                    let value = (0..col).fold(matrix_mut_get(b, row, col), |value, inner| {
                        value
                            - matrix_mut_get(b, row, inner)
                                * triangular_get(a, triangle, diagonal, transform, inner, col)
                    });
                    matrix_set(
                        b,
                        row,
                        col,
                        value / triangular_get(a, triangle, diagonal, transform, col, col),
                    );
                }
            }
        }
    }
}

#[test]
fn provider_dispatches_reductions_affine_updates_and_real_complex_transforms() {
    let provider = CblasProvider::new();
    let mut context = provider.context();
    let left = Vector::from_vec(vec![1.0_f64, 2.0, 3.0]);
    let right = Vector::from_vec(vec![4.0_f64, 5.0, 6.0]);
    assert_eq!(
        left.view()
            .dotu_scoped(
                &right,
                &scioxide_linalg::LinalgExecutionOptions::new(),
                context.reborrow()
            )
            .unwrap(),
        32.0
    );
    assert_eq!(
        left.view()
            .norm2_scoped(
                &scioxide_linalg::LinalgExecutionOptions::new(),
                context.reborrow()
            )
            .unwrap(),
        14.0_f64.sqrt()
    );

    let complex_left = Vector::from_vec(vec![Complex64::new(1.0, 2.0)]);
    let complex_right = Vector::from_vec(vec![Complex64::new(3.0, 4.0)]);
    assert_eq!(
        complex_left
            .view()
            .dotc_left_scoped(
                &complex_right,
                &scioxide_linalg::LinalgExecutionOptions::new(),
                context.reborrow()
            )
            .unwrap(),
        Complex64::new(11.0, -2.0)
    );

    let binary32 = Vector::from_vec(vec![1.0_f32, 2.0, 3.0]);
    assert_eq!(
        binary32
            .view()
            .dotu_scoped(
                &binary32,
                &scioxide_linalg::LinalgExecutionOptions::new(),
                context.reborrow()
            )
            .unwrap(),
        14.0
    );
    let complex32 = Vector::from_vec(vec![Complex32::new(1.0, -2.0)]);
    assert_eq!(
        complex32
            .view()
            .dotc_left_scoped(
                &complex32,
                &scioxide_linalg::LinalgExecutionOptions::new(),
                context.reborrow()
            )
            .unwrap(),
        Complex32::new(5.0, 0.0)
    );

    let mut output = [99.0_f64; 3];
    let mut scratch = [0.0_f64; 3];
    left.view()
        .axpby_into(
            2.0,
            right.view(),
            1.0,
            &scioxide_linalg::LinalgExecutionOptions::new(),
            VectorViewMut::from_slice(&mut output),
            &mut scioxide_linalg::LinalgWorkspace::from_scalar_slice(&mut scratch),
            context.reborrow(),
        )
        .unwrap();
    assert_eq!(output, [6.0, 9.0, 12.0]);

    let mut copied = Vector::from_vec(vec![0.0; 3]);
    left.view()
        .copy_into(
            scioxide_linalg::CopyPolicy::RequireFinite,
            &scioxide_linalg::LinalgExecutionOptions::new(),
            copied.view_mut(),
            &mut scioxide_linalg::LinalgWorkspace::from_scalar_slice(&mut scratch),
            context.reborrow(),
        )
        .unwrap();
    scioxide_linalg::VectorViewMut::scale_in_place(
        3.0,
        &scioxide_linalg::LinalgExecutionOptions::new(),
        copied.view_mut(),
        &mut scioxide_linalg::LinalgWorkspace::from_scalar_slice(&mut scratch),
        context.reborrow(),
    )
    .unwrap();
    assert_eq!(copied.as_slice(), &[3.0, 6.0, 9.0]);
    assert_eq!(provider.calls.load(Ordering::Relaxed), 8);
    assert_eq!(provider.operations.load(Ordering::Relaxed) & 0b111, 0b111);
}

#[test]
fn gemv_accepts_row_column_and_strided_vectors() {
    let provider = CblasProvider::new();
    let mut context = provider.context();
    let row_major = Matrix::from_vec(2, 3, vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let input_data = [2.0_f64, 0.0, 3.0, 0.0, 4.0];
    let input = VectorView::from_slice_with_stride(&input_data, 3, 2).unwrap();
    let mut output = [0.0; 2];
    let mut scratch = [f64::NAN; 2];
    row_major
        .view()
        .mul_vec_into(
            input,
            &scioxide_linalg::LinalgExecutionOptions::new(),
            VectorViewMut::from_slice(&mut output),
            &mut scioxide_linalg::LinalgWorkspace::from_scalar_slice(&mut scratch),
            context.reborrow(),
        )
        .unwrap();
    assert_eq!(output, [20.0, 47.0]);

    let column_view = row_major.view().transpose();
    let mut column_output = [0.0; 3];
    let mut column_scratch = [0.0; 3];
    column_view
        .mul_vec_into(
            VectorView::from_slice(&[2.0, 3.0]),
            &scioxide_linalg::LinalgExecutionOptions::new(),
            VectorViewMut::from_slice(&mut column_output),
            &mut scioxide_linalg::LinalgWorkspace::from_scalar_slice(&mut column_scratch),
            context.reborrow(),
        )
        .unwrap();
    assert_eq!(column_output, [14.0, 19.0, 24.0]);
}

#[test]
fn transpose_and_adjoint_use_declared_transform_encodings() {
    let capabilities = BlasCapabilities::empty(BlasIntegerWidth::Lp64)
        .with_layout(BlasLayout::RowMajor)
        .with_transform(BlasTransform::None)
        .with_transform(BlasTransform::Transpose)
        .with_transform(BlasTransform::Adjoint)
        .with_operation(ScalarKind::F64, BlasOperation::Gemv, BlasUlp::new(4))
        .unwrap()
        .with_operation(ScalarKind::Complex64, BlasOperation::Gemv, BlasUlp::new(4))
        .unwrap()
        .with_operation(ScalarKind::F64, BlasOperation::Gemm, BlasUlp::new(4))
        .unwrap()
        .with_operation(ScalarKind::Complex64, BlasOperation::Gemm, BlasUlp::new(4))
        .unwrap();
    let provider = CblasProvider::with_capabilities(capabilities);
    let mut context = provider.context();

    let real = Matrix::from_vec(2, 3, vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let mut output = [0.0; 3];
    let mut scratch = [0.0; 3];
    real.view()
        .transpose()
        .mul_vec_into(
            VectorView::from_slice(&[2.0, 3.0]),
            &scioxide_linalg::LinalgExecutionOptions::new(),
            VectorViewMut::from_slice(&mut output),
            &mut scioxide_linalg::LinalgWorkspace::from_scalar_slice(&mut scratch),
            context.reborrow(),
        )
        .unwrap();
    assert_eq!(output, [14.0, 19.0, 24.0]);

    let complex = Matrix::from_vec(
        2,
        2,
        vec![
            Complex64::new(1.0, 2.0),
            Complex64::new(-3.0, 1.0),
            Complex64::new(0.5, -1.0),
            Complex64::new(2.0, 0.0),
        ],
    )
    .unwrap();
    let identity = Matrix::<Complex64>::identity(2).unwrap();
    let expected = complex.view().adjoint().try_to_matrix().unwrap();
    let mut actual = Matrix::zeros(2, 2).unwrap();
    let mut workspace = LinalgWorkspace::new();
    complex
        .view()
        .adjoint()
        .mul_mat_into(
            identity.view(),
            &scioxide_linalg::LinalgExecutionOptions::new()
                .with_execution_policy(ExecutionPolicy::Sequential),
            actual.view_mut(),
            &mut workspace,
            context.reborrow(),
        )
        .unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn missing_borrowed_provider_packing_rejects_before_dispatch() {
    let provider = CblasProvider::new();
    let matrix = Matrix::<f64>::identity(2).unwrap();
    let options = scioxide_linalg::LinalgExecutionOptions::new();
    let requirements = LinalgWorkspace::<f64>::workspace_requirements(
        &scioxide_linalg::DenseWorkspaceRequest::product(2, 2, 2, &options),
    )
    .unwrap();
    let mut scratch = [0.0; 4];
    let mut workspace = LinalgWorkspace::from_scalar_slice(&mut scratch);
    workspace.try_reserve(&requirements).unwrap();
    let mut output = Matrix::from_vec(2, 2, vec![7.0; 4]).unwrap();
    let error = matrix
        .view()
        .mul_mat_into(
            &matrix,
            &options,
            output.view_mut(),
            &mut workspace,
            provider.context(),
        )
        .unwrap_err();
    assert_eq!(
        error.kind(),
        scioxide_core::SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(provider.calls.load(Ordering::Relaxed), 0);
    assert_eq!(output.as_slice(), &[7.0; 4]);
}

#[test]
fn gemm_packs_general_strides_and_commits_once() {
    let provider = CblasProvider::new();
    let mut context = provider.context();
    let left_data = [1.0_f64, 90.0, 2.0, 3.0, 90.0, 4.0];
    let left = MatrixView::from_slice_with_strides(2, 2, &left_data, 3, 2).unwrap();
    let right = Matrix::from_vec(2, 2, vec![5.0, 6.0, 7.0, 8.0]).unwrap();
    let mut output = Matrix::from_vec(2, 2, vec![99.0; 4]).unwrap();
    let mut workspace = LinalgWorkspace::new();
    left.mul_mat_into(
        right.view(),
        &scioxide_linalg::LinalgExecutionOptions::new()
            .with_execution_policy(ExecutionPolicy::Sequential),
        output.view_mut(),
        &mut workspace,
        context.reborrow(),
    )
    .unwrap();
    assert_eq!(output.as_slice(), &[19.0, 22.0, 43.0, 50.0]);
    let requirements = LinalgWorkspace::<f64>::workspace_requirements(
        &scioxide_linalg::DenseWorkspaceRequest::product(
            2,
            2,
            2,
            &scioxide_linalg::LinalgExecutionOptions::new(),
        )
        .with_provider_packing(),
    )
    .unwrap();
    assert_eq!(
        workspace.check(&requirements).unwrap(),
        scioxide_core::Capacity::Sufficient
    );

    provider.fail_after_write.store(true, Ordering::Relaxed);
    let sentinel = output.clone();
    let error = left
        .mul_mat_into(
            right.view(),
            &scioxide_linalg::LinalgExecutionOptions::new()
                .with_execution_policy(ExecutionPolicy::Sequential),
            output.view_mut(),
            &mut workspace,
            context.reborrow(),
        )
        .unwrap_err();
    assert!(matches!(
        (error).view(),
        scioxide_core::SciOxideErrorView::CapabilityFailure {
            capability: Capability::Blas,
            ..
        }
    ));
    assert_eq!(output, sentinel);
}

#[test]
fn beta_zero_provider_paths_do_not_read_prior_candidate_values() {
    let provider = CblasProvider::new();
    let mut context = provider.context();
    let left = Matrix::from_vec(1, 1, vec![2.0_f64]).unwrap();
    let right = Matrix::from_vec(1, 1, vec![3.0_f64]).unwrap();
    let mut output = Matrix::from_vec(1, 1, vec![f64::NAN]).unwrap();
    let mut workspace = LinalgWorkspace::new();
    left.view()
        .mul_mat_into(
            &right,
            &scioxide_linalg::LinalgExecutionOptions::new()
                .with_execution_policy(ExecutionPolicy::Sequential),
            output.view_mut(),
            &mut workspace,
            context.reborrow(),
        )
        .unwrap();
    assert_eq!(output.as_slice(), &[6.0]);
}

#[test]
fn provider_matches_reference_within_its_declared_ulp_contract() {
    let provider = CblasProvider::new();
    let mut context = provider.context();
    let left = Matrix::from_vec(2, 3, vec![0.5_f64, -2.0, 3.0, 4.0, 1.5, -0.25]).unwrap();
    let right = Matrix::from_vec(3, 2, vec![2.0_f64, 1.0, -1.0, 0.5, 3.0, -4.0]).unwrap();
    let mut reference = Matrix::zeros(2, 2).unwrap();
    left.view()
        .mul_mat_into(
            &right,
            &scioxide_linalg::LinalgExecutionOptions::new()
                .with_execution_policy(ExecutionPolicy::Sequential)
                .with_accuracy(scioxide_core::Accuracy::Exact),
            reference.view_mut(),
            &mut LinalgWorkspace::new(),
            LinalgExecutionContext::sequential(),
        )
        .unwrap();
    let mut provided = Matrix::zeros(2, 2).unwrap();
    let mut workspace = LinalgWorkspace::new();
    left.view()
        .mul_mat_into(
            &right,
            &scioxide_linalg::LinalgExecutionOptions::new()
                .with_execution_policy(ExecutionPolicy::Sequential),
            provided.view_mut(),
            &mut workspace,
            context.reborrow(),
        )
        .unwrap();

    let maximum = provider
        .capabilities()
        .operation_ulp(ScalarKind::F64, BlasOperation::Gemm)
        .unwrap()
        .maximum();
    reference
        .as_slice()
        .iter()
        .zip(provided.as_slice())
        .enumerate()
        .for_each(|(index, (&expected, &actual))| {
            assert!(f64_ulp_distance(expected, actual) <= u64::from(maximum));
            let magnitude = (0..3)
                .map(|k| {
                    left.view().get(index / 2, k).unwrap().abs()
                        * right.view().get(k, index % 2).unwrap().abs()
                })
                .sum::<f64>();
            let gamma = 3.0 * f64::EPSILON / (1.0 - 3.0 * f64::EPSILON);
            assert!((expected - actual).abs() <= gamma * magnitude + 3.0 * f64::from_bits(1));
        });
    let first = provided.clone();
    left.view()
        .mul_mat_into(
            &right,
            &scioxide_linalg::LinalgExecutionOptions::new()
                .with_execution_policy(ExecutionPolicy::Sequential),
            provided.view_mut(),
            &mut workspace,
            context.reborrow(),
        )
        .unwrap();
    first
        .as_slice()
        .iter()
        .zip(provided.as_slice())
        .for_each(|(&expected, &actual)| assert_eq!(expected.to_bits(), actual.to_bits()));

    let complex = Matrix::from_vec(
        2,
        2,
        vec![
            Complex64::new(1.0, 2.0),
            Complex64::new(-3.0, 1.0),
            Complex64::new(0.5, -1.0),
            Complex64::new(2.0, 0.0),
        ],
    )
    .unwrap();
    let identity = Matrix::<Complex64>::identity(2).unwrap();
    let expected = complex.view().adjoint().try_to_matrix().unwrap();
    let mut actual = Matrix::zeros(2, 2).unwrap();
    let mut complex_workspace = LinalgWorkspace::new();
    complex
        .view()
        .adjoint()
        .mul_mat_into(
            identity.view(),
            &scioxide_linalg::LinalgExecutionOptions::new()
                .with_execution_policy(ExecutionPolicy::Sequential),
            actual.view_mut(),
            &mut complex_workspace,
            context.reborrow(),
        )
        .unwrap();
    assert_eq!(actual, expected);
}

fn f64_ulp_distance(left: f64, right: f64) -> u64 {
    fn ordered(value: f64) -> u64 {
        let bits = value.to_bits();
        if bits >> 63 == 0 {
            bits | (1_u64 << 63)
        } else {
            !bits
        }
    }
    ordered(left).abs_diff(ordered(right))
}

struct EmptyProvider {
    capabilities: BlasCapabilities,
}

impl BlasProvider for EmptyProvider {
    fn name(&self) -> &'static str {
        "empty"
    }
    fn capabilities(&self) -> BlasCapabilities {
        self.capabilities.clone()
    }
    fn execute_f32(&self, _: BlasCall<'_, f32>, _: NonZeroUsize) -> Result<(), CapabilityError> {
        Err(CapabilityError::new("unexpected call"))
    }
    fn execute_f64(&self, _: BlasCall<'_, f64>, _: NonZeroUsize) -> Result<(), CapabilityError> {
        Err(CapabilityError::new("unexpected call"))
    }
    fn execute_complex32(
        &self,
        _: BlasCall<'_, Complex32>,
        _: NonZeroUsize,
    ) -> Result<(), CapabilityError> {
        Err(CapabilityError::new("unexpected call"))
    }
    fn execute_complex64(
        &self,
        _: BlasCall<'_, Complex64>,
        _: NonZeroUsize,
    ) -> Result<(), CapabilityError> {
        Err(CapabilityError::new("unexpected call"))
    }
}

#[test]
fn unsupported_alignment_and_integer_width_obey_explicit_fallback() {
    let alignment = 1_usize << (usize::BITS - 1);
    let strict_alignment = EmptyProvider {
        capabilities: BlasCapabilities::empty(BlasIntegerWidth::Lp64)
            .with_layout(BlasLayout::RowMajor)
            .with_transform(BlasTransform::None)
            .with_operation(ScalarKind::F64, BlasOperation::Dotu, BlasUlp::new(0))
            .unwrap()
            .with_minimum_alignment(alignment),
    };
    let values = Vector::from_vec(vec![1.0_f64]);
    let error = values
        .view()
        .dotu_scoped(
            &values,
            &scioxide_linalg::LinalgExecutionOptions::new(),
            LinalgExecutionContext::sequential().with_blas(BlasDispatch::new(
                &strict_alignment,
                BlasFallbackPolicy::Error,
            )),
        )
        .unwrap_err();
    assert!(matches!(
        (error).view(),
        scioxide_core::SciOxideErrorView::CapabilityUnavailable {
            capability: Capability::Blas,
            ..
        }
    ));

    let reference = values
        .view()
        .dotu_scoped(
            &values,
            &scioxide_linalg::LinalgExecutionOptions::new(),
            LinalgExecutionContext::sequential().with_blas(BlasDispatch::new(
                &strict_alignment,
                BlasFallbackPolicy::Reference,
            )),
        )
        .unwrap();
    assert_eq!(reference, 1.0);

    // Zero work runs natively even when its stride exceeds the provider width.
    let lp64 = CblasProvider::new();
    let empty = VectorView::<f64>::from_slice_with_stride(&[], 0, i32::MAX as usize + 1).unwrap();
    let dot = empty
        .dotu_scoped(
            empty,
            &scioxide_linalg::LinalgExecutionOptions::new(),
            lp64.context(),
        )
        .unwrap();
    assert_eq!(dot.to_bits(), 0.0_f64.to_bits());
    assert_eq!(lp64.calls.load(Ordering::Relaxed), 0);

    let axpy_provider = CblasProvider::new();
    let left = Vector::from_vec(vec![f64::MAX]);
    let right = Vector::from_vec(vec![f64::MAX]);
    let mut output = Vector::from_vec(vec![7.0]);
    let mut scratch = [0.0];
    let error = left
        .view()
        .axpby_into(
            -2.0,
            &right,
            2.0,
            &scioxide_linalg::LinalgExecutionOptions::new(),
            output.view_mut(),
            &mut scioxide_linalg::LinalgWorkspace::from_scalar_slice(&mut scratch),
            axpy_provider.context(),
        )
        .unwrap_err();
    assert!(matches!(
        (error).view(),
        scioxide_core::SciOxideErrorView::CapabilityUnavailable {
            capability: Capability::Blas,
            ..
        }
    ));
    assert_eq!(output.as_slice(), &[7.0]);

    left.view()
        .axpby_into(
            -2.0,
            &right,
            2.0,
            &scioxide_linalg::LinalgExecutionOptions::new(),
            output.view_mut(),
            &mut scioxide_linalg::LinalgWorkspace::from_scalar_slice(&mut scratch),
            LinalgExecutionContext::sequential().with_blas(BlasDispatch::new(
                &axpy_provider,
                BlasFallbackPolicy::Reference,
            )),
        )
        .unwrap();
    assert_eq!(output.as_slice(), &[0.0]);
}

#[test]
fn malformed_provider_results_are_typed_failures() {
    let provider = CblasProvider::new();
    provider.malformed_result.store(true, Ordering::Relaxed);
    let values = Vector::from_vec(vec![1.0_f64, 2.0]);
    let error = values
        .view()
        .dotu_scoped(
            &values,
            &scioxide_linalg::LinalgExecutionOptions::new(),
            provider.context(),
        )
        .unwrap_err();
    assert!(matches!(
        (error).view(),
        scioxide_core::SciOxideErrorView::NonFinite {
            origin: NonFiniteOrigin::Callback,
            ..
        }
    ));
}

struct BudgetExecutor;

impl Executor for BudgetExecutor {
    fn worker_budget(&self) -> NonZeroUsize {
        NonZeroUsize::new(4).unwrap()
    }

    fn execute(
        &self,
        context: ExecutionContext<'_>,
        plan: ExecutionPlan,
        operation: &PartitionFn<'_>,
    ) -> Result<(), SciOxideError> {
        SequentialExecutor.execute(context, plan, operation)
    }
}

#[test]
fn provider_receives_outer_and_nested_worker_budgets() {
    let provider = CblasProvider::new();
    let executor = BudgetExecutor;
    let execution = ExecutionContext::new(&executor);
    let values = Vector::from_vec(vec![2.0_f64]);
    values
        .view()
        .dotu_scoped(
            &values,
            &scioxide_linalg::LinalgExecutionOptions::new(),
            LinalgExecutionContext::new(execution)
                .with_blas(BlasDispatch::new(&provider, BlasFallbackPolicy::Error)),
        )
        .unwrap();
    assert_eq!(provider.workers.load(Ordering::Relaxed), 4);

    let left = Matrix::from_vec(200, 1, vec![1.0_f64; 200]).unwrap();
    let right = Matrix::from_vec(1, 1, vec![2.0_f64]).unwrap();
    let mut output = Matrix::zeros(200, 1).unwrap();
    let mut workspace = LinalgWorkspace::new();
    left.view()
        .mul_mat_into(
            &right,
            &scioxide_linalg::LinalgExecutionOptions::new()
                .with_execution_policy(ExecutionPolicy::Sequential),
            output.view_mut(),
            &mut workspace,
            LinalgExecutionContext::new(execution)
                .with_blas(BlasDispatch::new(&provider, BlasFallbackPolicy::Error)),
        )
        .unwrap();
    assert_eq!(provider.workers.load(Ordering::Relaxed), 1);
    left.view()
        .mul_mat_into(
            &right,
            &scioxide_linalg::LinalgExecutionOptions::new()
                .with_execution_policy(ExecutionPolicy::Parallel(NonZeroUsize::new(2).unwrap())),
            output.view_mut(),
            &mut workspace,
            LinalgExecutionContext::new(execution)
                .with_blas(BlasDispatch::new(&provider, BlasFallbackPolicy::Error)),
        )
        .unwrap();
    assert_eq!(provider.workers.load(Ordering::Relaxed), 2);

    let plan = execution.plan(
        ExecutionPolicy::Parallel(NonZeroUsize::new(2).unwrap()),
        1,
        scioxide_core::PlanLimits {
            work_per_item: core::num::NonZeroUsize::MIN,
            auto_threshold: NonZeroUsize::MIN,
            worker_limit: NonZeroUsize::new(2).unwrap(),
        },
    );
    execution
        .execute(plan, &|_, (), scope| {
            values.view().dotu_scoped(
                &values,
                &scioxide_linalg::LinalgExecutionOptions::new(),
                LinalgExecutionContext::new(scope)
                    .with_blas(BlasDispatch::new(&provider, BlasFallbackPolicy::Error)),
            )?;
            Ok(())
        })
        .unwrap();
    assert_eq!(provider.workers.load(Ordering::Relaxed), 1);

    let mut single_threaded = CblasProvider::new();
    single_threaded.capabilities = single_threaded
        .capabilities
        .clone()
        .with_threading(BlasThreading::SingleThreaded);
    values
        .view()
        .dotu_scoped(
            &values,
            &scioxide_linalg::LinalgExecutionOptions::new(),
            LinalgExecutionContext::new(execution).with_blas(BlasDispatch::new(
                &single_threaded,
                BlasFallbackPolicy::Error,
            )),
        )
        .unwrap();
    assert_eq!(single_threaded.workers.load(Ordering::Relaxed), 1);
}

#[test]
fn exact_accuracy_requires_zero_ulp_or_explicit_reference_fallback() {
    let provider = CblasProvider::new();
    let x = [1.0e16_f64, 1.0, -1.0e16];
    let y = [1.0; 3];
    let exact = provider.context();
    assert!(matches!(
        VectorView::from_slice(&x)
            .dotu_scoped(
                VectorView::from_slice(&y),
                &scioxide_linalg::LinalgExecutionOptions::new()
                    .with_accuracy(scioxide_core::Accuracy::Exact),
                exact
            )
            .unwrap_err()
            .view(),
        scioxide_core::SciOxideErrorView::CapabilityUnavailable { .. }
    ));
    assert_eq!(provider.calls.load(Ordering::Relaxed), 0);
    let fallback = LinalgExecutionContext::sequential()
        .with_blas(BlasDispatch::new(&provider, BlasFallbackPolicy::Reference));
    assert_eq!(
        VectorView::from_slice(&x)
            .dotu_scoped(
                VectorView::from_slice(&y),
                &scioxide_linalg::LinalgExecutionOptions::new()
                    .with_accuracy(scioxide_core::Accuracy::Exact),
                fallback
            )
            .unwrap(),
        1.0
    );
    assert_eq!(provider.calls.load(Ordering::Relaxed), 0);
    let provider = CblasProvider::with_capabilities(
        provider
            .capabilities
            .with_operation(ScalarKind::F64, BlasOperation::Dotu, BlasUlp::new(0))
            .unwrap(),
    );
    assert_eq!(
        VectorView::from_slice(&[2.0_f64])
            .dotu_scoped(
                VectorView::from_slice(&[3.0]),
                &scioxide_linalg::LinalgExecutionOptions::new()
                    .with_accuracy(scioxide_core::Accuracy::Exact),
                provider.context()
            )
            .unwrap(),
        6.0
    );
    assert_eq!(provider.calls.load(Ordering::Relaxed), 1);
}

#[test]
fn principal_matrix_functions_retain_provider_and_publish_atomically() {
    use scioxide_linalg::{
        MatrixFunction, MatrixFunctionOptions, MatrixFunctionPlan, MatrixFunctionWorkspaceRequest,
    };
    let a = Matrix::from_vec(2, 2, vec![2.0_f64, 0.3, 0., 2.]).unwrap();
    for function in [
        MatrixFunction::Exp,
        MatrixFunction::Log,
        MatrixFunction::Sqrt,
    ] {
        let provider = CblasProvider::new();
        let plan = MatrixFunctionPlan::new(&a, function, &MatrixFunctionOptions::new()).unwrap();
        let mut workspace = plan
            .workspace(&MatrixFunctionWorkspaceRequest::Evaluate)
            .unwrap();
        let mut output = Matrix::<f64>::zeros(2, 2).unwrap();
        plan.evaluate_into(output.view_mut(), &mut workspace, provider.context())
            .unwrap();
        let (diagonal, off_diagonal) = match function {
            MatrixFunction::Exp => (2_f64.exp(), 0.3 * 2_f64.exp()),
            MatrixFunction::Log => (2_f64.ln(), 0.15),
            _ => (2_f64.sqrt(), 0.3 / (2. * 2_f64.sqrt())),
        };
        assert!((output.as_slice()[0] - diagonal).abs() < 512. * f64::EPSILON * diagonal);
        assert!((output.as_slice()[1] - off_diagonal).abs() < 512. * f64::EPSILON);
        assert!(provider.calls.load(Ordering::Relaxed) > 0);
        let before = output.clone();
        provider.fail_after_write.store(true, Ordering::Relaxed);
        assert!(
            plan.evaluate_into(output.view_mut(), &mut workspace, provider.context())
                .is_err()
        );
        assert_eq!(output, before);
    }
}

#[test]
fn invalid_alignment_is_rejected_before_provider_calls_under_both_policies() {
    for alignment in [0, 3, 6] {
        for fallback in [BlasFallbackPolicy::Error, BlasFallbackPolicy::Reference] {
            let provider = CblasProvider::with_capabilities(
                CblasProvider::new()
                    .capabilities()
                    .with_minimum_alignment(alignment),
            );
            assert_eq!(provider.capabilities().minimum_alignment(), alignment);
            let context = || {
                LinalgExecutionContext::sequential()
                    .with_blas(BlasDispatch::new(&provider, fallback))
            };
            let values = VectorView::from_slice(&[1.0_f64, 2.0]);
            let error = values
                .dotu_scoped(
                    values,
                    &scioxide_linalg::LinalgExecutionOptions::new(),
                    context(),
                )
                .unwrap_err();
            assert!(
                matches!(error.view(), scioxide_core::SciOxideErrorView::InvalidInput { argument, reason: scioxide_core::InvalidInputReason::OutOfRange, .. } if argument == scioxide_core::argument!("minimum_alignment"))
            );
            let mut output = Matrix::from_vec(1, 1, vec![17.0_f64]).unwrap();
            let matrix = MatrixView::from_slice(1, 1, &[2.0]).unwrap();
            let error = matrix
                .mul_mat_into(
                    matrix,
                    &LinalgExecutionOptions::new(),
                    output.view_mut(),
                    &mut LinalgWorkspace::new(),
                    context(),
                )
                .unwrap_err();
            assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::InvalidInput);
            assert_eq!(output.as_slice(), &[17.0]);
            assert_eq!(provider.calls.load(Ordering::Relaxed), 0);
        }
    }
}

fn flip_triangle(triangle: scioxide_core::Triangle) -> scioxide_core::Triangle {
    match triangle {
        scioxide_core::Triangle::Lower => scioxide_core::Triangle::Upper,
        scioxide_core::Triangle::Upper => scioxide_core::Triangle::Lower,
    }
}
fn packed_index(n: usize, triangle: scioxide_core::Triangle, row: usize, col: usize) -> usize {
    match triangle {
        scioxide_core::Triangle::Upper => col * (col + 1) / 2 + row,
        scioxide_core::Triangle::Lower => col * n - col * col.saturating_sub(1) / 2 + row - col,
    }
}
fn structured_shape<S: NumericScalar>(
    a: Operand<'_, S>,
) -> (usize, usize, scioxide_core::Triangle) {
    match a {
        Operand::Dense { matrix, triangle } => (matrix.rows(), matrix.cols(), triangle),
        Operand::Band {
            matrix: a,
            triangle,
        } => (a.rows(), a.cols(), triangle.unwrap_or(Triangle::Lower)),
        Operand::Packed(a) => (a.order(), a.order(), a.triangle()),
    }
}
fn structured_get<S: NumericScalar>(a: Operand<'_, S>, row: usize, col: usize) -> S {
    match a {
        Operand::Dense { matrix, .. } => matrix_get(matrix, row, col),
        Operand::Packed(a) => a.values()[packed_index(a.order(), a.triangle(), row, col)],
        Operand::Band { matrix: a, .. } => {
            if row > col && row - col > a.lower() || col > row && col - row > a.upper() {
                S::ZERO
            } else {
                a.values()[col * a.leading_dimension() + a.upper() + row - col]
            }
        }
    }
}

#[path = "../blas_provider/conformance.rs"]
mod conformance;

// The reference BLAS modified-Givens construction, including its rescaling
// and flag-dependent parameter storage. This provider never calls SciOxide kernels.
fn cblas_rotmg<R: RealScalar>(d1: &mut R, d2: &mut R, x: &mut R, y: R) -> [R; 5] {
    let mut flag;
    let (mut h11, mut h21, mut h12, mut h22) = (R::ZERO, R::ZERO, R::ZERO, R::ZERO);
    let p2 = *d2 * y;
    if *d1 < R::ZERO {
        flag = -R::ONE;
        *d1 = R::ZERO;
        *d2 = R::ZERO;
        *x = R::ZERO;
    } else if p2 == R::ZERO {
        return [-R::TWO, R::ZERO, R::ZERO, R::ZERO, R::ZERO];
    } else {
        let p1 = *d1 * *x;
        let q1 = p1 * *x;
        let q2 = p2 * y;
        if q1.abs() > q2.abs() {
            h21 = -y / *x;
            h12 = p2 / p1;
            let u = R::ONE - h12 * h21;
            if u > R::ZERO {
                flag = R::ZERO;
                *d1 /= u;
                *d2 /= u;
                *x *= u;
            } else {
                flag = -R::ONE;
                *d1 = R::ZERO;
                *d2 = R::ZERO;
                *x = R::ZERO;
                h21 = R::ZERO;
                h12 = R::ZERO;
            }
        } else if q2 < R::ZERO {
            flag = -R::ONE;
            *d1 = R::ZERO;
            *d2 = R::ZERO;
            *x = R::ZERO;
        } else {
            flag = R::ONE;
            h11 = p1 / p2;
            h22 = *x / y;
            let u = R::ONE + h11 * h22;
            let next = *d2 / u;
            *d2 = *d1 / u;
            *d1 = next;
            *x = y * u;
        }
    }
    let gam = R::try_from_f64_exact(4096.0).unwrap();
    let square = gam * gam;
    let reciprocal = R::ONE / square;
    if *d1 != R::ZERO {
        while *d1 <= reciprocal || *d1 >= square {
            if flag == R::ZERO {
                h11 = R::ONE;
                h22 = R::ONE;
            } else if flag == R::ONE {
                h21 = -R::ONE;
                h12 = R::ONE;
            }
            flag = -R::ONE;
            if *d1 <= reciprocal {
                *d1 *= square;
                *x /= gam;
                h11 /= gam;
                h12 /= gam;
            } else {
                *d1 /= square;
                *x *= gam;
                h11 *= gam;
                h12 *= gam;
            }
        }
    }
    if *d2 != R::ZERO {
        while d2.abs() <= reciprocal || d2.abs() >= square {
            if flag == R::ZERO {
                h11 = R::ONE;
                h22 = R::ONE;
            } else if flag == R::ONE {
                h21 = -R::ONE;
                h12 = R::ONE;
            }
            flag = -R::ONE;
            if d2.abs() <= reciprocal {
                *d2 *= square;
                h21 /= gam;
                h22 /= gam;
            } else {
                *d2 /= square;
                h21 *= gam;
                h22 *= gam;
            }
        }
    }
    if flag == R::ZERO {
        [flag, R::ZERO, h21, h12, R::ZERO]
    } else if flag == R::ONE {
        [flag, h11, R::ZERO, R::ZERO, h22]
    } else {
        [flag, h11, h21, h12, h22]
    }
}

#[path = "../blas_provider/faults.rs"]
mod faults;

#[test]
fn dense_krylov_applications_use_the_injected_provider_and_preserve_output_on_failure() {
    use scioxide_core::{Budget, ResidualTolerance};
    use scioxide_linalg::{CgSolver, IterativeWorkspaceRequest, Preconditioning};
    let matrix =
        Matrix::from_vec(3, 3, vec![4.0_f64, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0]).unwrap();
    let operator = matrix.view().try_finite().unwrap();
    let rhs = Vector::from_vec(vec![1.0, 2.0, 3.0]);
    let solver = CgSolver::new(
        ResidualTolerance::new(0.0, 1e-12).unwrap(),
        Budget::try_from(16).unwrap(),
    );
    let mut workspace = solver
        .workspace(&IterativeWorkspaceRequest::operator(&operator).unwrap())
        .unwrap();
    let provider = CblasProvider::new();
    let mut output = Vector::zeros(3).unwrap();
    let report = solver
        .solve_into(
            &operator,
            &rhs,
            Preconditioning::None,
            output.view_mut(),
            &mut workspace,
            provider.context(),
        )
        .unwrap();
    assert!(report.is_success());
    assert!(provider.calls.load(Ordering::Relaxed) >= 3);
    let input = Matrix::from_vec(3, 2, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]).unwrap();
    let mut block = Matrix::zeros(3, 2).unwrap();
    let options = LinalgExecutionOptions::new();
    let mut block_workspace = LinalgWorkspace::workspace(
        &scioxide_linalg::OperatorWorkspaceRequest::block(&operator, 2, &options).unwrap(),
    )
    .unwrap();
    let calls = provider.calls.load(Ordering::Relaxed);
    scioxide_linalg::apply_operator_matrix_into(
        &operator,
        &input,
        &options,
        block.view_mut(),
        &mut block_workspace,
        provider.context(),
    )
    .unwrap();
    assert_eq!(provider.calls.load(Ordering::Relaxed) - calls, 2);
    for (&actual, expected) in block
        .as_slice()
        .iter()
        .zip([6.0, 21.0, 10.0, 25.0, 8.0, 17.0])
    {
        assert!((actual - expected).abs() < 1e-14);
    }
    provider.fail_after_write.store(true, Ordering::Relaxed);
    let before = output.clone();
    assert!(
        solver
            .solve_into(
                &operator,
                &rhs,
                Preconditioning::None,
                output.view_mut(),
                &mut workspace,
                provider.context()
            )
            .is_err()
    );
    assert_eq!(output, before);
}

#[test]
fn iterative_solves_honour_the_composed_execution_policy() {
    use scioxide_core::{Budget, ResidualTolerance};
    use scioxide_linalg::{CgSolver, IterativeWorkspaceRequest, Preconditioning};
    let matrix =
        Matrix::from_vec(3, 3, vec![4.0_f64, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0]).unwrap();
    let operator = matrix.view().try_finite().unwrap();
    let rhs = Vector::from_vec(vec![1.0, 2.0, 3.0]);
    let provider = CblasProvider::new();
    let executor = BudgetExecutor;
    for (policy, workers) in [
        (ExecutionPolicy::Auto, 4),
        (ExecutionPolicy::Sequential, 1),
        (ExecutionPolicy::Parallel(NonZeroUsize::new(2).unwrap()), 2),
    ] {
        let solver = CgSolver::new(
            ResidualTolerance::new(0.0, 1e-12).unwrap(),
            Budget::try_from(16).unwrap(),
        )
        .with_execution_options(LinalgExecutionOptions::new().with_execution_policy(policy));
        let mut workspace = solver
            .workspace(&IterativeWorkspaceRequest::operator(&operator).unwrap())
            .unwrap();
        let mut output = Vector::zeros(3).unwrap();
        provider.workers.store(0, Ordering::Relaxed);
        let report = solver
            .solve_into(
                &operator,
                &rhs,
                Preconditioning::None,
                output.view_mut(),
                &mut workspace,
                LinalgExecutionContext::new(ExecutionContext::new(&executor))
                    .with_blas(BlasDispatch::new(&provider, BlasFallbackPolicy::Error)),
            )
            .unwrap();
        assert!(report.is_success(), "{policy:?}");
        assert_eq!(
            provider.workers.load(Ordering::Relaxed),
            workers,
            "{policy:?}"
        );
    }
}

#[test]
fn vector_and_triangular_vector_calls_follow_the_composed_policy() {
    let provider = CblasProvider::new();
    let executor = BudgetExecutor;
    let values = Vector::from_vec(vec![2.0_f64, -1.0, 3.0]);
    let triangular =
        Matrix::from_vec(3, 3, vec![2.0, 0.0, 0.0, 1.0, 3.0, 0.0, 1.0, 1.0, 4.0]).unwrap();
    for (policy, workers) in [
        (ExecutionPolicy::Auto, 4),
        (ExecutionPolicy::Sequential, 1),
        (ExecutionPolicy::Parallel(NonZeroUsize::new(2).unwrap()), 2),
    ] {
        let options = LinalgExecutionOptions::new().with_execution_policy(policy);
        let context = || {
            LinalgExecutionContext::new(ExecutionContext::new(&executor))
                .with_blas(BlasDispatch::new(&provider, BlasFallbackPolicy::Error))
        };
        let observed = |name: &str| {
            assert_eq!(
                provider.workers.swap(0, Ordering::Relaxed),
                workers,
                "{name} {policy:?}"
            );
        };
        values
            .view()
            .dotu_scoped(&values, &options, context())
            .unwrap();
        observed("dotu");
        values.view().norm2_scoped(&options, context()).unwrap();
        observed("norm2");
        values.view().asum_scoped(&options, context()).unwrap();
        observed("asum");
        values.view().iamax_scoped(&options, context()).unwrap();
        observed("iamax");
        let plan = scioxide_linalg::TriangularPlan::from_dense(
            triangular.view(),
            Triangle::Lower,
            scioxide_core::DiagonalKind::Explicit,
        )
        .unwrap();
        let mut x = Vector::zeros(3).unwrap();
        let report = plan
            .solve_into(
                StructuredOperation::Normal,
                &values,
                &scioxide_linalg::DirectSolveOptions::new(scioxide_linalg::RankPolicy::Exact)
                    .with_execution_options(options),
                x.view_mut(),
                &mut plan
                    .workspace(&scioxide_linalg::StructuredWorkspaceRequest::solve_matrix(
                        3, 1, &options,
                    ))
                    .unwrap(),
                context(),
            )
            .unwrap();
        assert!(report.termination().is_success());
        observed("trsv");
    }
}

#[test]
fn dense_rank_updates_stage_in_the_provider_layout() {
    let capabilities = [BlasOperation::Geru, BlasOperation::Syr, BlasOperation::Syr2]
        .into_iter()
        .fold(
            BlasCapabilities::empty(BlasIntegerWidth::Lp64)
                .with_layout(BlasLayout::ColumnMajor)
                .with_transform(BlasTransform::None),
            |caps, operation| {
                caps.with_operation(ScalarKind::F64, operation, BlasUlp::new(0))
                    .unwrap()
            },
        );
    let provider = CblasProvider::with_capabilities(capabilities);
    let options = LinalgExecutionOptions::new();
    let x = [2.0, -1.0];
    let y = [3.0, 4.0, 0.5];
    let mut general = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    scioxide_linalg::MatrixViewMut::rank_one_update_in_place(
        0.5,
        VectorView::from_slice(&x),
        VectorView::from_slice(&y),
        &options,
        scioxide_linalg::MatrixViewMut::from_slice(2, 3, &mut general).unwrap(),
        &mut LinalgWorkspace::new(),
        provider.context(),
    )
    .unwrap();
    assert_eq!(general, [4.0, 6.0, 3.5, 2.5, 3.0, 5.75]);
    let mut symmetric = [1.0_f64, 9.0, 2.0, 3.0];
    scioxide_linalg::MatrixViewMut::symmetric_rank_one_update_in_place(
        Triangle::Lower,
        1.0,
        VectorView::from_slice(&x),
        &options,
        scioxide_linalg::MatrixViewMut::from_slice(2, 2, &mut symmetric).unwrap(),
        &mut LinalgWorkspace::new(),
        provider.context(),
    )
    .unwrap();
    assert_eq!(symmetric, [5.0, 9.0, 0.0, 4.0]);
    scioxide_linalg::MatrixViewMut::symmetric_rank_two_update_in_place(
        Triangle::Upper,
        1.0,
        VectorView::from_slice(&x),
        VectorView::from_slice(&[1.0, 1.0]),
        &options,
        scioxide_linalg::MatrixViewMut::from_slice(2, 2, &mut symmetric).unwrap(),
        &mut LinalgWorkspace::new(),
        provider.context(),
    )
    .unwrap();
    assert_eq!(symmetric, [9.0, 10.0, 0.0, 2.0]);
    assert_eq!(provider.calls.load(Ordering::Relaxed), 3);
}

#[test]
fn both_triangles_of_a_wider_band_reach_the_provider() {
    // Column-major tridiagonal storage; the padding lies outside the band.
    let data = [0.0_f64, 2.0, 1.0, 5.0, 3.0, 1.0, 6.0, 4.0, 0.0];
    let band = BandedMatrixView::new(3, 3, 1, 1, 3, &data).unwrap();
    let options = LinalgExecutionOptions::new();
    for (triangle, rhs) in [
        (Triangle::Lower, [2.0, 4.0, 9.0]),
        (Triangle::Upper, [7.0, 9.0, 4.0]),
    ] {
        let provider = CblasProvider::new();
        let plan = scioxide_linalg::TriangularPlan::from_band(
            band,
            triangle,
            scioxide_core::DiagonalKind::Explicit,
        )
        .unwrap();
        let mut x = Vector::zeros(3).unwrap();
        let report = plan
            .solve_into(
                StructuredOperation::Normal,
                &rhs[..],
                &scioxide_linalg::DirectSolveOptions::new(scioxide_linalg::RankPolicy::Exact)
                    .with_execution_options(options),
                x.view_mut(),
                &mut plan
                    .workspace(&scioxide_linalg::StructuredWorkspaceRequest::solve_matrix(
                        3, 1, &options,
                    ))
                    .unwrap(),
                provider.context(),
            )
            .unwrap();
        assert!(report.termination().is_success());
        let expected = if triangle == Triangle::Lower {
            [1.0, 1.0, 2.0]
        } else {
            [1.0, 1.0, 1.0]
        };
        for (actual, expected) in x.as_slice().iter().zip(expected) {
            assert!(
                (actual - expected).abs() <= 4.0 * f64::EPSILON,
                "{triangle:?}"
            );
        }
        assert_eq!(provider.calls.load(Ordering::Relaxed), 1, "{triangle:?}");
    }
}

#[test]
fn real_conjugated_vectors_need_no_conjugate_capability() {
    let capabilities = BlasCapabilities::empty(BlasIntegerWidth::Lp64)
        .with_layout(BlasLayout::RowMajor)
        .with_transform(BlasTransform::None)
        .with_operation(ScalarKind::F64, BlasOperation::Dotu, BlasUlp::new(0))
        .unwrap();
    let provider = CblasProvider::with_capabilities(capabilities);
    let values = [1.0_f64, 2.0, 3.0];
    let view = VectorView::from_slice(&values).conjugate();
    let dot = view
        .dotu_scoped(view, &LinalgExecutionOptions::new(), provider.context())
        .unwrap();
    assert_eq!(dot, 14.0);
    assert_eq!(provider.calls.load(Ordering::Relaxed), 1);
}

#[test]
fn zero_work_makes_no_provider_call() {
    let provider = CblasProvider::new();
    let options = LinalgExecutionOptions::new();
    let empty = VectorView::<f64>::from_slice(&[]);
    let zero = 0.0_f64.to_bits();
    assert_eq!(
        empty
            .dotu_scoped(empty, &options, provider.context())
            .unwrap()
            .to_bits(),
        zero
    );
    assert_eq!(
        empty
            .norm2_scoped(&options, provider.context())
            .unwrap()
            .to_bits(),
        zero
    );
    assert_eq!(
        empty
            .asum_scoped(&options, provider.context())
            .unwrap()
            .to_bits(),
        zero
    );
    let mut nothing: [f64; 0] = [];
    VectorViewMut::axpby_in_place(
        2.0,
        empty,
        1.0,
        &options,
        VectorViewMut::from_slice(&mut nothing),
        &mut LinalgWorkspace::new(),
        provider.context(),
    )
    .unwrap();
    VectorViewMut::scale_in_place(
        2.0,
        &options,
        VectorViewMut::from_slice(&mut nothing),
        &mut LinalgWorkspace::new(),
        provider.context(),
    )
    .unwrap();
    // A 0 x 3 GEMV has no output rows.
    MatrixView::<f64>::from_slice(0, 3, &[])
        .unwrap()
        .mul_vec_into(
            VectorView::from_slice(&[1.0, 2.0, 3.0]),
            &options,
            VectorViewMut::from_slice(&mut nothing),
            &mut LinalgWorkspace::new(),
            provider.context(),
        )
        .unwrap();
    assert_eq!(provider.calls.load(Ordering::Relaxed), 0);
}
