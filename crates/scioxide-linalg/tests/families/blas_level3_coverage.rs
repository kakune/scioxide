#![allow(clippy::unwrap_used)]

use scioxide_core::{Complex, ExecutionPolicy};
use scioxide_core::{DiagonalKind, Triangle};
use scioxide_linalg::{
    LinalgExecutionContext, LinalgWorkspace, MatrixSide, MatrixView, MatrixViewMut,
    StructuredOperation,
};

#[test]
fn symmetric_product_handles_both_sides_and_ignored_triangle() {
    let stored = [2.0_f64, f64::NAN, 1.0, 3.0];
    let general = [1.0, 2.0, 3.0, 4.0];
    let a = MatrixView::from_slice(2, 2, &stored).unwrap();
    let b = MatrixView::from_slice(2, 2, &general).unwrap();
    let mut left = [0.0; 4];
    let mut right = [0.0; 4];
    let mut scratch = LinalgWorkspace::new();
    a.symmetric_mul_mat_in_place(
        b,
        MatrixSide::Left,
        Triangle::Lower,
        1.0,
        0.0,
        &scioxide_linalg::LinalgExecutionOptions::new(),
        MatrixViewMut::from_slice(2, 2, &mut left).unwrap(),
        &mut scratch,
        LinalgExecutionContext::sequential(),
    )
    .unwrap();
    a.symmetric_mul_mat_in_place(
        b,
        MatrixSide::Right,
        Triangle::Lower,
        1.0,
        0.0,
        &scioxide_linalg::LinalgExecutionOptions::new(),
        MatrixViewMut::from_slice(2, 2, &mut right).unwrap(),
        &mut scratch,
        LinalgExecutionContext::sequential(),
    )
    .unwrap();
    assert_eq!(left, [5.0, 8.0, 10.0, 14.0]);
    assert_eq!(right, [4.0, 7.0, 10.0, 15.0]);
}

#[test]
fn hermitian_rank_two_k_keeps_diagonal_real_and_other_triangle() {
    let a_data = [Complex::new(1.0_f64, 1.0), Complex::new(2.0, 0.0)];
    let b_data = [Complex::new(3.0_f64, 0.0), Complex::new(0.0, 1.0)];
    let a = MatrixView::from_slice(2, 1, &a_data).unwrap();
    let b = MatrixView::from_slice(2, 1, &b_data).unwrap();
    let untouched = Complex::new(99.0, 77.0);
    let mut out = [
        Complex::new(0.0, 0.0),
        untouched,
        Complex::new(0.0, 0.0),
        Complex::new(0.0, 0.0),
    ];
    let mut scratch = LinalgWorkspace::new();
    scioxide_linalg::MatrixViewMut::hermitian_rank_two_k_in_place(
        a,
        b,
        Triangle::Lower,
        StructuredOperation::Normal,
        Complex::new(1.0, 0.0),
        Complex::new(0.0, 0.0),
        &scioxide_linalg::LinalgExecutionOptions::new(),
        MatrixViewMut::from_slice(2, 2, &mut out).unwrap(),
        &mut scratch,
        LinalgExecutionContext::sequential(),
    )
    .unwrap();
    assert_eq!(out[0], Complex::new(6.0, 0.0));
    assert_eq!(out[1], untouched);
    assert_eq!(out[2], Complex::new(7.0, 1.0));
    assert_eq!(out[3], Complex::new(0.0, 0.0));
}

#[test]
fn triangular_solve_is_transactional_and_handles_right_side() {
    let triangular = [2.0_f64, 1.0, f64::NAN, 3.0];
    let t = MatrixView::from_slice(2, 2, &triangular).unwrap();
    let mut b = [4.0, 7.0, 8.0, 11.0];
    let mut workspace = LinalgWorkspace::new();
    t.triangular_solve_mat_in_place(
        MatrixSide::Right,
        Triangle::Upper,
        DiagonalKind::Explicit,
        StructuredOperation::Normal,
        1.0,
        &scioxide_linalg::LinalgExecutionOptions::new(),
        MatrixViewMut::from_slice(2, 2, &mut b).unwrap(),
        &mut workspace,
        LinalgExecutionContext::sequential(),
    )
    .unwrap();
    assert_eq!(b, [2.0, 5.0 / 3.0, 4.0, 7.0 / 3.0]);
    let before = b;
    let singular = [0.0_f64, 1.0, f64::NAN, 3.0];
    let t = MatrixView::from_slice(2, 2, &singular).unwrap();
    assert!(
        t.triangular_solve_mat_in_place(
            MatrixSide::Right,
            Triangle::Upper,
            DiagonalKind::Explicit,
            StructuredOperation::Normal,
            1.0,
            &scioxide_linalg::LinalgExecutionOptions::new(),
            MatrixViewMut::from_slice(2, 2, &mut b).unwrap(),
            &mut workspace,
            LinalgExecutionContext::sequential()
        )
        .is_err()
    );
    assert_eq!(b, before);
}

#[test]
fn shared_rank_k_is_public_and_checked() {
    let a_data = [1.0_f64, 2.0, 3.0, 4.0];
    let a = MatrixView::from_slice(2, 2, &a_data).unwrap();
    let mut out = [0.0; 4];
    let mut workspace = LinalgWorkspace::new();
    scioxide_linalg::MatrixViewMut::symmetric_rank_k_in_place(
        a,
        Triangle::Lower,
        StructuredOperation::Normal,
        1.0,
        0.0,
        &scioxide_linalg::LinalgExecutionOptions::new()
            .with_execution_policy(ExecutionPolicy::Sequential),
        MatrixViewMut::from_slice(2, 2, &mut out).unwrap(),
        &mut workspace,
        LinalgExecutionContext::sequential(),
    )
    .unwrap();
    assert_eq!(out[0], 5.0);
    assert_eq!(out[2], 11.0);
    assert_eq!(out[3], 25.0);
}

#[test]
fn triangular_products_preserve_finite_triples_and_mixed_exponents() {
    let high = f64::from_bits(1923_u64 << 52);
    let low = f64::from_bits(123_u64 << 52);
    for accuracy in [
        scioxide_core::Accuracy::Standard,
        scioxide_core::Accuracy::Exact,
    ] {
        for (order, a, b, alpha, expected) in [
            (1, vec![high], vec![low], high, vec![high]),
            (
                2,
                vec![high, low, f64::NAN, 1.0],
                vec![low, high],
                1.0,
                vec![2.0, high],
            ),
        ] {
            let matrix = MatrixView::from_slice(order, order, &a).unwrap();
            let mut output = b;
            matrix
                .triangular_mul_mat_in_place(
                    MatrixSide::Left,
                    Triangle::Upper,
                    DiagonalKind::Explicit,
                    StructuredOperation::Normal,
                    alpha,
                    &scioxide_linalg::LinalgExecutionOptions::new().with_accuracy(accuracy),
                    MatrixViewMut::from_slice(order, 1, &mut output).unwrap(),
                    &mut LinalgWorkspace::new(),
                    LinalgExecutionContext::sequential(),
                )
                .unwrap();
            for (actual, expected) in output.iter().zip(expected) {
                assert!((actual - expected).abs() <= 8.0 * f64::EPSILON * expected.abs());
            }
        }
    }
}

#[test]
fn native_triangular_product_uses_the_requested_executor_and_scalar_scratch() {
    use core::{
        num::NonZeroUsize,
        sync::atomic::{AtomicUsize, Ordering},
    };
    use scioxide_core::{
        ExecutionContext, ExecutionPlan, Executor, PartitionFn, SciOxideError, SequentialExecutor,
    };
    struct RecordingExecutor(AtomicUsize);
    impl Executor for RecordingExecutor {
        fn worker_budget(&self) -> NonZeroUsize {
            NonZeroUsize::new(4).unwrap()
        }
        fn execute(
            &self,
            context: ExecutionContext<'_>,
            plan: ExecutionPlan,
            operation: &PartitionFn<'_>,
        ) -> Result<(), SciOxideError> {
            self.0.fetch_add(plan.worker_count(), Ordering::Relaxed);
            SequentialExecutor.execute(context, plan, operation)
        }
    }
    let n = 257;
    let executor = RecordingExecutor(AtomicUsize::new(0));
    let a = scioxide_linalg::Matrix::<f64>::identity(n).unwrap();
    let options = scioxide_linalg::LinalgExecutionOptions::new()
        .with_execution_policy(ExecutionPolicy::Parallel(executor.worker_budget()));
    let requirements = LinalgWorkspace::<f64>::workspace_requirements(
        &scioxide_linalg::DenseWorkspaceRequest::triangular_product(n, n, 2, &options),
    )
    .unwrap();
    assert_eq!(requirements.scalar_len().unwrap(), 4 * n);
    let mut scratch = vec![0.0; requirements.scalar_len().unwrap()];
    let mut workspace = LinalgWorkspace::from_scalar_slice(&mut scratch);
    workspace.try_reserve(&requirements).unwrap();
    let mut output = vec![2.0; 2 * n];
    a.view()
        .triangular_mul_mat_in_place(
            MatrixSide::Left,
            Triangle::Lower,
            DiagonalKind::Explicit,
            StructuredOperation::Normal,
            1.0,
            &options,
            MatrixViewMut::from_slice(n, 2, &mut output).unwrap(),
            &mut workspace,
            ExecutionContext::new(&executor),
        )
        .unwrap();
    assert!(output.iter().all(|&value| (value - 2.0).abs() < 1e-14));
    assert!(executor.0.load(Ordering::Relaxed) > 1);
}
