//! Accuracy, layout, and allocation contracts for the shared sparse kernels.
#![allow(clippy::unwrap_used)]
use scioxide_core::{
    Accuracy, AlgebraicScalar, BlockVectorView, BlockVectorViewMut, ExecutionContext,
    LinearOperator, NumericScalar, ProductSum, RealScalar,
};
use scioxide_sparse::{
    DiaMatrix, DuplicatePolicy, ExplicitZeroPolicy, SparseOperation, SparseProductOptions,
    SparseProductWorkspace, sparse_dense_into,
};

fn reference<S: NumericScalar>(
    dense: &[S],
    rows: usize,
    cols: usize,
    x: &[S],
    op: SparseOperation,
) -> Vec<S> {
    let (m, n) = if op == SparseOperation::Normal {
        (rows, cols)
    } else {
        (cols, rows)
    };
    (0..m)
        .map(|r| {
            let mut sum = ProductSum::new();
            for (c, &x) in x[..n].iter().enumerate() {
                let a = match op {
                    SparseOperation::Normal => dense[r * cols + c],
                    SparseOperation::Transpose => dense[c * cols + r],
                    SparseOperation::Adjoint => dense[c * cols + r].conj(),
                };
                sum.add_product(a, x).unwrap();
            }
            sum.finish().unwrap()
        })
        .collect()
}
fn check<S: NumericScalar>(actual: &[S], expected: &[S], accuracy: Accuracy) {
    for (&a, &b) in actual.iter().zip(expected) {
        if accuracy == Accuracy::Exact {
            assert_eq!(
                a.real_component().decompose(),
                b.real_component().decompose()
            );
            assert_eq!(
                a.imaginary_component().decompose(),
                b.imaginary_component().decompose()
            );
        } else {
            let bound = S::Real::try_from_f64_lossy(256.).unwrap()
                * S::Real::EPSILON
                * (S::Real::ONE + b.abs());
            assert!((a - b).abs() <= bound, "{a:?} != {b:?}");
        }
    }
}

#[test]
fn diagonal_active_ranges_and_underflow_retries() {
    for (rows, cols) in [(19, 3), (3, 19), (7, 9)] {
        let ri: Vec<_> = (0..rows)
            .flat_map(|r| core::iter::repeat_n(r, cols))
            .collect();
        let ci: Vec<_> = (0..rows).flat_map(|_| 0..cols).collect();
        let values: Vec<_> = ri
            .iter()
            .zip(&ci)
            .map(|(&r, &c)| ((r + c) % 7) as f64 / 13.)
            .collect();
        let coo = {
            let mut builder = scioxide_sparse::SparseBuilder::new(rows, cols);
            builder.extend(&ri, &ci, &values).and_then(|()| {
                builder.finish_coo(DuplicatePolicy::Sum, ExplicitZeroPolicy::Preserve)
            })
        }
        .unwrap();
        let dia = DiaMatrix::from_sparse(coo.view()).unwrap();
        for op in [
            SparseOperation::Normal,
            SparseOperation::Transpose,
            SparseOperation::Adjoint,
        ] {
            let (m, n) = if op == SparseOperation::Normal {
                (rows, cols)
            } else {
                (cols, rows)
            };
            let x: Vec<_> = (0..n).map(|i| (i as f64 - 3.) / 11.).collect();
            for accuracy in [Accuracy::Standard, Accuracy::Exact] {
                let mut y = vec![0.; m];
                sparse_dense_into(
                    dia.view(),
                    op,
                    BlockVectorView::new(&x, n, 1).unwrap(),
                    SparseProductOptions::new().with_accuracy(accuracy),
                    &mut BlockVectorViewMut::new(&mut y, m, 1).unwrap(),
                    &mut dense_workspace(dia.view(), op, 1),
                    ExecutionContext::sequential(),
                )
                .unwrap();
                check(&y, &reference(&values, rows, cols, &x, op), accuracy);
            }
        }
    }
    let tiny = f64::from_bits(1);
    let coo = {
        let mut builder = scioxide_sparse::SparseBuilder::new(1, 2);
        builder
            .extend(&[0, 0], &[0, 1], &[tiny, tiny])
            .and_then(|()| builder.finish_coo(DuplicatePolicy::Sum, ExplicitZeroPolicy::Preserve))
    }
    .unwrap();
    for accuracy in [Accuracy::Standard, Accuracy::Exact] {
        let mut y = [0.];
        coo.view()
            .with_accuracy(accuracy)
            .apply_into(&[0.5, 0.5], &mut y, &mut [0.])
            .unwrap();
        assert_eq!(y[0].to_bits(), tiny.to_bits());
    }
}
#[test]
fn derived_exact_products_and_triangular_rounding_boundaries() {
    use scioxide_core::{DiagonalKind, Triangle};
    use scioxide_sparse::{JacobiPreconditioner, SparseArithmeticOptions, SparseTriangular};
    let a = {
        let mut builder = scioxide_sparse::SparseBuilder::new(3, 3);
        builder
            .extend(
                &[0, 1, 1, 2, 2, 2],
                &[0, 0, 1, 0, 1, 2],
                &[3.0_f64, 1. / 7., 5., -1. / 9., 2. / 11., 7.],
            )
            .and_then(|()| builder.finish_csr(DuplicatePolicy::Sum, ExplicitZeroPolicy::Preserve))
    }
    .unwrap();
    let rhs = [1. / 3., 1. / 5., 1. / 7.];
    let mut legacy = [0.; 3];
    for row in 0..3 {
        let mut sum = ProductSum::new();
        sum.add_product(rhs[row], 1.).unwrap();
        let mut pivot = 0.;
        for p in a.row_ptr()[row]..a.row_ptr()[row + 1] {
            let col = a.col_indices()[p];
            if col == row {
                pivot = a.values()[p];
            } else {
                sum.subtract_product(a.values()[p], legacy[col]).unwrap();
            }
        }
        legacy[row] = sum.finish().unwrap() / pivot;
    }
    let plan = SparseTriangular::new(a.clone(), Triangle::Lower, DiagonalKind::Explicit).unwrap();
    for accuracy in [Accuracy::Standard, Accuracy::Exact] {
        let mut y = [0.; 3];
        plan.solve_into(
            &rhs,
            scioxide_sparse::SparseOperation::Normal,
            scioxide_sparse::SparseSolveOptions::new().with_accuracy(accuracy),
            &mut y,
            &mut scioxide_sparse::SparseSolveWorkspace::borrowed(&mut [0.; 6]),
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap();
        check(&y, &legacy, accuracy);
        let jacobi = JacobiPreconditioner::new(&a).unwrap();
        jacobi
            .apply_into(
                &rhs,
                scioxide_sparse::SparseSolveOptions::new().with_accuracy(accuracy),
                &mut y,
                &mut scioxide_sparse::SparseSolveWorkspace::borrowed(&mut [0.; 3]),
                scioxide_core::ExecutionContext::sequential(),
            )
            .unwrap();
        for ((&actual, &b), &d) in y.iter().zip(&rhs).zip(jacobi.inverse_diagonal()) {
            let mut sum = ProductSum::new();
            sum.add_product(b, d).unwrap();
            check(&[actual], &[sum.finish().unwrap()], accuracy);
        }
    }
    let options = SparseArithmeticOptions::new().with_accuracy(Accuracy::Exact);
    let sum = a.add_with_options(&a, options).unwrap();
    let product = a.hadamard_with_options(&a, options).unwrap();
    for ((&actual, &square), &v) in sum.values().iter().zip(product.values()).zip(a.values()) {
        let mut reference = ProductSum::new();
        reference.add_product(v, v).unwrap();
        assert_eq!(actual.to_bits(), (v + v).to_bits());
        assert_eq!(square.to_bits(), reference.finish().unwrap().to_bits());
    }
}

#[test]
fn empty_block_does_not_walk_unaddressable_column_extents() {
    let matrix = {
        let mut builder = scioxide_sparse::SparseBuilder::<f64>::new(0, 0);
        builder
            .extend(&[], &[], &[])
            .and_then(|()| builder.finish_coo(DuplicatePolicy::Sum, ExplicitZeroPolicy::Preserve))
    }
    .unwrap();
    sparse_dense_into(
        matrix.view(),
        SparseOperation::Normal,
        BlockVectorView::new(&[], 0, usize::MAX).unwrap(),
        SparseProductOptions::default(),
        &mut BlockVectorViewMut::new(&mut [], 0, usize::MAX).unwrap(),
        &mut dense_workspace(matrix.view(), SparseOperation::Normal, usize::MAX),
        ExecutionContext::sequential(),
    )
    .unwrap();
}
fn dense_workspace<S: NumericScalar>(
    matrix: scioxide_sparse::SparseView<'_, S>,
    operation: SparseOperation,
    right_hand_sides: usize,
) -> SparseProductWorkspace<S> {
    let mut workspace = SparseProductWorkspace::new();
    workspace
        .try_reserve(
            &scioxide_sparse::SparseProductRequirements::sparse_dense(
                matrix,
                operation,
                right_hand_sides,
            )
            .unwrap(),
        )
        .unwrap();
    workspace
}
