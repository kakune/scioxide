//! Independent storage and arithmetic contracts.
#![allow(clippy::unwrap_used)]
use scioxide_core::{AdjointOperator, LinearOperator};
use scioxide_sparse::{
    BsrMatrix, CsrMatrix, DiaMatrix, DuplicatePolicy, ExplicitZeroPolicy, SparseBuilder,
    SparseProductPlanOptions,
};
#[test]
fn rectangular_formats_and_structural_expansion() {
    let coo = {
        let mut builder = scioxide_sparse::SparseBuilder::new(4, 6);
        builder
            .extend(&[0, 1, 3], &[1, 1, 5], &[2.0_f64, 0.0, 3.0])
            .and_then(|()| builder.finish_coo(DuplicatePolicy::Sum, ExplicitZeroPolicy::Preserve))
    }
    .unwrap();
    let bsr = BsrMatrix::from_sparse(coo.view(), 2, 3).unwrap();
    assert_eq!(bsr.block_count(), 2);
    assert_eq!(
        BsrMatrix::conversion_requirements(coo.view(), 2, 3)
            .unwrap()
            .values(),
        12
    );
    let dia = DiaMatrix::from_sparse(coo.view()).unwrap();
    assert_eq!(dia.offsets(), &[0, 1, 2]);
    for view in [coo.view(), bsr.view(), dia.view()] {
        let mut output = [9.0; 4];
        view.apply_into(&[1.0; 6], &mut output, &mut [0.0; 4])
            .unwrap();
        assert_eq!(output, [2.0, 0.0, 0.0, 3.0]);
        let mut adjoint = [9.0; 6];
        view.apply_adjoint_into(&[1.0; 4], &mut adjoint, &mut [0.0; 6])
            .unwrap();
        assert_eq!(adjoint, [0.0, 2.0, 0.0, 0.0, 0.0, 3.0]);
        assert_eq!(view.to_csr(ExplicitZeroPolicy::Drop).unwrap().nnz(), 2);
    }
    assert_eq!(bsr.transpose().unwrap().transpose().unwrap(), bsr);
    assert_eq!(dia.transpose().unwrap().transpose().unwrap(), dia);
}
#[test]
fn malformed_and_empty_storage() {
    assert!(BsrMatrix::<f32>::from_raw(3, 4, 2, 2, vec![0], vec![], vec![]).is_err());
    assert!(
        BsrMatrix::<f32>::from_raw(2, 2, 1, 1, vec![0, 2, 2], vec![0, 0], vec![1.0, 2.0]).is_err()
    );
    assert!(DiaMatrix::from_raw(2, 3, vec![1], vec![1.0_f32, 2.0, 3.0]).is_err());
    assert!(DiaMatrix::<f32>::from_raw(0, 3, vec![0], vec![0.0; 3]).is_err());
    assert!(DiaMatrix::<f32>::from_raw(2, 2, vec![isize::MIN], vec![0.0; 2]).is_err());
    let empty = BsrMatrix::<f32>::from_raw(0, 6, 2, 3, vec![0], vec![], vec![]).unwrap();
    assert_eq!(empty.view().iter().count(), 0);
    assert!(BsrMatrix::from_sparse(empty.view(), usize::MAX, 2).is_err());
    let extreme = DiaMatrix::from_raw(
        isize::MIN.unsigned_abs() + 1,
        1,
        vec![isize::MIN],
        vec![1.0_f32],
    )
    .unwrap();
    assert_eq!(DiaMatrix::from_sparse(extreme.view()).unwrap(), extreme);
}
#[test]
fn builder_duplicates_and_transactional_application() {
    let mut builder = SparseBuilder::new(2, 2);
    builder.push(0, 0, f64::MAX).unwrap();
    builder.push(0, 0, -f64::MAX).unwrap();
    builder.push(1, 1, 2.0).unwrap();
    let matrix = BsrMatrix::from_sparse(
        builder
            .finish_coo(DuplicatePolicy::Sum, ExplicitZeroPolicy::Preserve)
            .unwrap()
            .view(),
        2,
        2,
    )
    .unwrap();
    assert_eq!(matrix.values(), &[0.0, 0.0, 0.0, 2.0]);
    let mut output = [7.0; 2];
    assert!(
        matrix
            .apply_into(&[1.0, f64::MAX], &mut output, &mut [0.0; 2])
            .is_err()
    );
    assert_eq!(output, [7.0; 2]);
}

use scioxide_core::{
    BlockVectorView, BlockVectorViewMut, Complex, ExecutionContext, NumericScalar, RealScalar,
    Tolerance,
};
use scioxide_sparse::{
    SparseOperation, SparseProductOptions, SparseProductPlan, SparseProductWorkspace,
    SparsePruning, sparse_block_diagonal, sparse_block_grid, sparse_dense_into, sparse_diagonals,
};
fn product_workspace<S: NumericScalar>(plan: &SparseProductPlan) -> SparseProductWorkspace<S> {
    // Two worker row states serve the parallel evaluations below.
    plan.workspace(SparseProductOptions::new().with_execution_policy(
        scioxide_core::ExecutionPolicy::Parallel(core::num::NonZeroUsize::new(2).unwrap()),
    ))
    .unwrap()
}
fn scalar<S: NumericScalar>(real: f64, imag: f64) -> S {
    S::from_components(
        S::Real::try_from_f64_exact(real).unwrap(),
        S::Real::try_from_f64_exact(imag).unwrap(),
    )
}
fn all_domains<S: NumericScalar>() {
    // Small integer-valued complex operands: ordinary independent dense loops
    // are exact here in both widths, without production ProductSum as oracle.
    let (m, k, n) = (4, 6, 2);
    let mut ar = Vec::new();
    let mut ac = Vec::new();
    let mut av = Vec::new();
    let mut br = Vec::new();
    let mut bc = Vec::new();
    let mut bv = Vec::new();
    let mut ad = vec![S::ZERO; m * k];
    let mut bd = vec![S::ZERO; k * n];
    for r in 0..m {
        for c in 0..k {
            if (r + c) % 3 == 0 {
                let v = scalar::<S>((r + 1) as f64, -((c % 2) as f64));
                ar.push(r);
                ac.push(c);
                av.push(v);
                ad[r * k + c] = v;
            }
        }
    }
    for r in 0..k {
        for c in 0..n {
            if (r + c) % 2 == 0 {
                let v = scalar::<S>((c + 1) as f64, (r % 3) as f64);
                br.push(r);
                bc.push(c);
                bv.push(v);
                bd[r * n + c] = v;
            }
        }
    }
    let a = {
        let mut builder = scioxide_sparse::SparseBuilder::new(m, k);
        builder
            .extend(&ar, &ac, &av)
            .and_then(|()| builder.finish_coo(DuplicatePolicy::Sum, ExplicitZeroPolicy::Preserve))
    }
    .unwrap();
    let a_csr = a.to_csr().unwrap();
    let a_csc = a.to_csc().unwrap();
    let a_bsr = BsrMatrix::from_sparse(a.view(), 2, 3).unwrap();
    let a_dia = DiaMatrix::from_sparse(a.view()).unwrap();
    let b = {
        let mut builder = scioxide_sparse::SparseBuilder::new(k, n);
        builder
            .extend(&br, &bc, &bv)
            .and_then(|()| builder.finish_csr(DuplicatePolicy::Sum, ExplicitZeroPolicy::Preserve))
    }
    .unwrap();
    let mut expected = vec![S::ZERO; m * n];
    for r in 0..m {
        for c in 0..n {
            for j in 0..k {
                expected[r * n + c] = expected[r * n + c] + ad[r * k + j] * bd[j * n + c];
            }
        }
    }
    let product = a_csr.multiply(&b).unwrap();
    let mut actual = vec![S::ZERO; m * n];
    product.to_dense_into(&mut actual).unwrap();
    assert_eq!(actual, expected);
    let plan = SparseProductPlan::new(&a_csr, &b).unwrap();
    let mut output = plan.create_output().unwrap();
    let mut scratch = product_workspace(&plan);
    plan.evaluate_into(
        &a_csr,
        &b,
        SparseProductOptions::default(),
        &mut output,
        &mut scratch,
        ExecutionContext::sequential(),
    )
    .unwrap();
    assert_eq!(output, product);
    #[cfg(feature = "std")]
    {
        use core::num::NonZeroUsize;
        use scioxide_core::{ExecutionPolicy, ThreadExecutor};
        let workers = NonZeroUsize::new(2).unwrap();
        let pool = ThreadExecutor::new(workers);
        plan.evaluate_into(
            &a_csr,
            &b,
            SparseProductOptions::new().with_execution_policy(ExecutionPolicy::Parallel(workers)),
            &mut output,
            &mut scratch,
            ExecutionContext::new(&pool),
        )
        .unwrap();
        assert_eq!(output, product);
    }
    for conjugate in [false, true] {
        let transposed = if conjugate {
            a_csr.adjoint().unwrap()
        } else {
            a_csr.transpose().unwrap()
        }
        .to_csr()
        .unwrap();
        let gram = transposed.multiply(&a_csr).unwrap();
        for r in 0..k {
            for c in 0..k {
                let expected = (0..m).fold(S::ZERO, |sum, i| {
                    let left = ad[i * k + r];
                    sum + (if conjugate { left.conj() } else { left }) * ad[i * k + c]
                });
                assert_eq!(gram.get(r, c).unwrap(), expected);
            }
        }
    }
    for view in [
        a.view(),
        a_csr.view(),
        a_csc.view(),
        a_bsr.view(),
        a_dia.view(),
    ] {
        for operation in [
            SparseOperation::Normal,
            SparseOperation::Transpose,
            SparseOperation::Adjoint,
        ] {
            let (rows, cols) = if operation == SparseOperation::Normal {
                (m, k)
            } else {
                (k, m)
            };
            let input: Vec<_> = (0..cols * 2)
                .map(|i| scalar::<S>((i % 4) as f64, (i % 3) as f64))
                .collect();
            let mut expected = vec![S::ZERO; rows * 2];
            for rhs in 0..2 {
                for r in 0..rows {
                    for c in 0..cols {
                        let v = match operation {
                            SparseOperation::Normal => ad[r * k + c],
                            SparseOperation::Transpose => ad[c * k + r],
                            SparseOperation::Adjoint => ad[c * k + r].conj(),
                        };
                        expected[rhs * rows + r] =
                            expected[rhs * rows + r] + v * input[rhs * cols + c];
                    }
                }
            }
            let mut out = vec![S::ZERO; rows * 2];
            sparse_dense_into(
                view,
                operation,
                BlockVectorView::new(&input, cols, 2).unwrap(),
                SparseProductOptions::new(),
                &mut BlockVectorViewMut::new(&mut out, rows, 2).unwrap(),
                &mut dense_workspace(view, operation, 2),
                ExecutionContext::sequential(),
            )
            .unwrap();
            assert_eq!(out, expected);
        }
    }
    let doubled = a_csr.add(&a_csr).unwrap();
    assert_eq!(doubled, a.view().scaled(scalar::<S>(2.0, 0.0)).unwrap());
    assert_eq!(
        a_csr
            .subtract_with_options(
                &a_csr,
                scioxide_sparse::SparseArithmeticOptions::new()
                    .with_zero_policy(ExplicitZeroPolicy::Drop)
            )
            .unwrap()
            .nnz(),
        0
    );
    let square = a_csr.hadamard(&a_csr).unwrap();
    for ((r, c, v), &original) in square.iter().zip(&av) {
        assert_eq!(v, original * original);
        assert_eq!(a_csr.get(r, c).unwrap(), original);
    }
}
#[test]
fn f32_products() {
    all_domains::<f32>();
}
#[test]
fn f64_products() {
    all_domains::<f64>();
}
#[test]
fn complex32_products() {
    all_domains::<Complex<f32>>();
}
#[test]
fn complex64_products() {
    all_domains::<Complex<f64>>();
}

#[test]
fn pruning_and_block_construction() {
    let diagonal =
        sparse_diagonals(3, 4, &[0, 1], &[&[1.0_f64, 2.0, 3.0], &[0.25, 0.5, 0.75]]).unwrap();
    let policy = SparsePruning::new(Tolerance::new(0.25, 0.25).unwrap(), 1.0).unwrap();
    let pruned = diagonal.view().pruned(policy).unwrap();
    assert_eq!(pruned.nnz(), 4);
    let dia = DiaMatrix::from_sparse(diagonal.view())
        .unwrap()
        .pruned(policy)
        .unwrap();
    assert_eq!(dia.to_csr(ExplicitZeroPolicy::Drop).unwrap(), pruned);
    let blocks = sparse_block_diagonal(&[diagonal.view(), diagonal.view()]).unwrap();
    assert_eq!((blocks.rows(), blocks.cols(), blocks.nnz()), (6, 8, 12));
    let grid = sparse_block_grid(
        &[3, 3],
        &[4, 4],
        &[Some(diagonal.view()), None, None, Some(diagonal.view())],
    )
    .unwrap();
    assert_eq!(grid, blocks);
    assert!(sparse_block_grid(&[2], &[4], &[Some(diagonal.view())]).is_err());
    let huge = {
        let mut builder = scioxide_sparse::SparseBuilder::new(1, 1);
        builder
            .extend(&[0], &[0], &[Complex::new(f64::MAX, f64::MAX)])
            .and_then(|()| builder.finish_coo(DuplicatePolicy::Sum, ExplicitZeroPolicy::Preserve))
    }
    .unwrap();
    let small = SparsePruning::new(Tolerance::new(f64::MAX, 0.0).unwrap(), 0.0).unwrap();
    assert_eq!(huge.view().pruned(small).unwrap().nnz(), 1);
    let large = SparsePruning::new(Tolerance::new(f64::MAX, 1.0).unwrap(), f64::MAX).unwrap();
    assert_eq!(huge.view().pruned(large).unwrap().nnz(), 0);
}
#[test]
fn product_admission_and_value_reuse() {
    let mut a = sparse_diagonals(2, 2, &[0], &[&[1.0_f64, 2.0]]).unwrap();
    let b = a.clone();
    let plan = SparseProductPlan::new(&a, &b).unwrap();
    let mut output = plan.create_output().unwrap();
    output
        .copy_values_from(&vec![7.0; output.values().len()])
        .unwrap();
    let before = output.clone();
    let mut scratch = product_workspace(&plan);
    let opts = SparseProductOptions::default();
    let ctx = ExecutionContext::sequential();
    assert!(
        plan.evaluate_into(
            &a,
            &b,
            opts,
            &mut output,
            &mut SparseProductWorkspace::default(),
            ctx
        )
        .is_err()
    );
    assert_eq!(output, before);
    a.set_value(1, f64::MAX).unwrap();
    assert!(
        plan.evaluate_into(&a, &b, opts, &mut output, &mut scratch, ctx)
            .is_err()
    );
    assert_eq!(output, before);
    assert_eq!(
        a.set_value(1, f64::NAN).unwrap_err().kind(),
        scioxide_core::SciOxideErrorKind::NonFinite
    );
    assert!(
        plan.evaluate_into(&a, &b, opts, &mut output, &mut scratch, ctx)
            .is_err()
    );
    assert_eq!(output, before);
    a.set_value(1, 3.0).unwrap();
    plan.evaluate_into(&a, &b, opts, &mut output, &mut scratch, ctx)
        .unwrap();
    assert_eq!(output.values(), &[1.0, 6.0]);
    let wrong = sparse_diagonals(2, 2, &[1], &[&[1.0_f64]]).unwrap();
    assert!(
        plan.evaluate_into(&wrong, &b, opts, &mut output, &mut scratch, ctx)
            .is_err()
    );
}
#[cfg(feature = "std")]
#[test]
fn injected_executor_matches_and_rolls_back() {
    use core::num::NonZeroUsize;
    use scioxide_core::{
        CancellationToken, ExecutionPlan, ExecutionPolicy, Executor, PartitionFn, SciOxideError,
        ThreadExecutor,
    };
    struct Incomplete;
    impl Executor for Incomplete {
        fn worker_budget(&self) -> NonZeroUsize {
            NonZeroUsize::new(2).unwrap()
        }
        fn execute(
            &self,
            _: ExecutionContext<'_>,
            _: ExecutionPlan,
            _: &PartitionFn<'_>,
        ) -> Result<(), SciOxideError> {
            Ok(())
        }
    }
    let a = sparse_diagonals(
        4,
        4,
        &[0, 1],
        &[&[1.0_f64, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0]],
    )
    .unwrap();
    let expected = a.multiply(&a).unwrap();
    let plan = SparseProductPlan::new(&a, &a).unwrap();
    let mut output = plan.create_output().unwrap();
    let mut scratch = product_workspace(&plan);
    let workers = NonZeroUsize::new(2).unwrap();
    let pool = ThreadExecutor::new(workers);
    let options =
        SparseProductOptions::new().with_execution_policy(ExecutionPolicy::Parallel(workers));
    plan.evaluate_into(
        &a,
        &a,
        options,
        &mut output,
        &mut scratch,
        ExecutionContext::new(&pool),
    )
    .unwrap();
    assert_eq!(output, expected);
    assert!(
        plan.evaluate_into(
            &a,
            &a,
            options,
            &mut output,
            &mut scratch,
            ExecutionContext::new(&Incomplete)
        )
        .is_err()
    );
    assert_eq!(output, expected);
    let token = CancellationToken::new();
    token.cancel();
    assert!(
        plan.evaluate_into(
            &a,
            &a,
            options,
            &mut output,
            &mut scratch,
            ExecutionContext::new(&pool).with_cancellation(&token)
        )
        .is_err()
    );
    assert_eq!(output, expected);
    let input = [1.0; 8];
    let mut dense = [0.0; 8];
    sparse_dense_into(
        a.view(),
        SparseOperation::Normal,
        BlockVectorView::new(&input, 4, 2).unwrap(),
        options,
        &mut BlockVectorViewMut::new(&mut dense, 4, 2).unwrap(),
        &mut dense_workspace(a.view(), SparseOperation::Normal, 2),
        ExecutionContext::new(&pool),
    )
    .unwrap();
    assert_eq!(dense, [6.0, 8.0, 10.0, 4.0, 6.0, 8.0, 10.0, 4.0]);
}
#[test]
fn zero_extents_and_sparse_large_shape() {
    let a = {
        let mut builder = scioxide_sparse::SparseBuilder::<f64>::new(3, 0);
        builder
            .extend(&[], &[], &[])
            .and_then(|()| builder.finish_csr(DuplicatePolicy::Sum, ExplicitZeroPolicy::Preserve))
    }
    .unwrap();
    let b = {
        let mut builder = scioxide_sparse::SparseBuilder::<f64>::new(0, 4);
        builder
            .extend(&[], &[], &[])
            .and_then(|()| builder.finish_csr(DuplicatePolicy::Sum, ExplicitZeroPolicy::Preserve))
    }
    .unwrap();
    let product = a.multiply(&b).unwrap();
    assert_eq!((product.rows(), product.cols(), product.nnz()), (3, 4, 0));
    // A dense intermediate would require 80 GB; compressed metadata stays small.
    let a = {
        let mut builder = scioxide_sparse::SparseBuilder::new(100_000, 100_000);
        builder
            .extend(&[0, 99_999], &[99_999, 0], &[2.0_f64, 3.0])
            .and_then(|()| builder.finish_csr(DuplicatePolicy::Sum, ExplicitZeroPolicy::Preserve))
    }
    .unwrap();
    let product = a.multiply(&a).unwrap();
    assert_eq!(product.nnz(), 2);
    assert_eq!(product.values(), &[6.0, 6.0]);
}

#[test]
fn exact_cancellation_and_subnormal_product() {
    let tiny = f64::from_bits(1);
    let a = {
        let mut builder = scioxide_sparse::SparseBuilder::new(1, 3);
        builder
            .extend(&[0, 0, 0], &[0, 1, 2], &[f64::MAX, f64::MAX, tiny])
            .and_then(|()| builder.finish_csr(DuplicatePolicy::Sum, ExplicitZeroPolicy::Preserve))
    }
    .unwrap();
    let b = {
        let mut builder = scioxide_sparse::SparseBuilder::new(3, 1);
        builder
            .extend(&[0, 1, 2], &[0, 0, 0], &[2.0, -2.0, 1.0])
            .and_then(|()| builder.finish_csr(DuplicatePolicy::Sum, ExplicitZeroPolicy::Preserve))
    }
    .unwrap();
    assert_eq!(a.multiply(&b).unwrap().values(), &[tiny]);
    let a = {
        let mut builder = scioxide_sparse::SparseBuilder::new(1, 2);
        builder
            .extend(&[0, 0], &[0, 1], &[1.0_f32, -1.0])
            .and_then(|()| builder.finish_csr(DuplicatePolicy::Sum, ExplicitZeroPolicy::Preserve))
    }
    .unwrap();
    let b = {
        let mut builder = scioxide_sparse::SparseBuilder::new(2, 1);
        builder
            .extend(&[0, 1], &[0, 0], &[1.0_f32, 1.0])
            .and_then(|()| builder.finish_csr(DuplicatePolicy::Sum, ExplicitZeroPolicy::Preserve))
    }
    .unwrap();
    assert_eq!(a.multiply(&b).unwrap().nnz(), 1);
    assert_eq!(
        a.multiply(&b)
            .unwrap()
            .view()
            .to_csr(ExplicitZeroPolicy::Drop)
            .unwrap()
            .nnz(),
        0
    );
}

#[test]
fn rejected_value_updates_preserve_conversion_invariants() {
    let mut coo = {
        let mut builder = scioxide_sparse::SparseBuilder::new(1, 1);
        builder
            .extend(&[0], &[0], &[1.0_f64])
            .and_then(|()| builder.finish_coo(DuplicatePolicy::Sum, ExplicitZeroPolicy::Preserve))
    }
    .unwrap();
    let mut csr = coo.to_csr().unwrap();
    let mut csc = coo.to_csc().unwrap();
    assert_eq!(
        coo.set_value(0, f64::INFINITY).unwrap_err().kind(),
        scioxide_core::SciOxideErrorKind::NonFinite
    );
    assert_eq!(
        csr.set_value(0, f64::NAN).unwrap_err().kind(),
        scioxide_core::SciOxideErrorKind::NonFinite
    );
    assert_eq!(
        csc.set_value(0, f64::NEG_INFINITY).unwrap_err().kind(),
        scioxide_core::SciOxideErrorKind::NonFinite
    );
    assert_eq!(coo.to_csr().unwrap(), csr);
    assert_eq!(coo.to_csc().unwrap(), csc);
    assert_eq!(csr.to_coo().unwrap(), coo);
    assert_eq!(csc.to_csr().unwrap(), csr);
    for matrix in [
        csr.transpose().unwrap().to_csr().unwrap(),
        csr.adjoint().unwrap().to_csr().unwrap(),
        csc.transpose().unwrap(),
        csc.adjoint().unwrap(),
    ] {
        assert_eq!(matrix.values(), &[1.0]);
    }
    let mut output = [7.0];
    csr.to_dense_into(&mut output).unwrap();
    assert_eq!(output, [1.0]);
    csc.to_dense_into(&mut output).unwrap();
    assert_eq!(output, [1.0]);
    BsrMatrix::from_sparse(coo.view(), 1, 1).unwrap();
    DiaMatrix::from_sparse(coo.view()).unwrap();
}

#[test]
fn bounded_products_report_exact_structural_requirements() {
    // Every left row discovers right columns in descending order, with duplicate
    // paths and explicit numerical zeros. Structural fill is still 2 * 4.
    let left = CsrMatrix::from_raw(
        2,
        4,
        vec![0, 4, 8],
        vec![0, 1, 2, 3, 0, 1, 2, 3],
        vec![0.0_f64; 8],
    )
    .unwrap();
    let right = CsrMatrix::from_raw(
        4,
        4,
        vec![0, 1, 3, 6, 10],
        vec![3, 2, 3, 1, 2, 3, 0, 1, 2, 3],
        vec![1.0; 10],
    )
    .unwrap();
    for limit in [0, 1, 4, 7] {
        let error = SparseProductPlan::with_options(
            &left,
            &right,
            SparseProductPlanOptions::new().with_max_output_entries(limit),
        )
        .unwrap_err();
        assert!(
            matches!(error.view(), scioxide_core::SciOxideErrorView::ResourceLimit { resource, required: 8, limit: capacity, .. } if capacity == limit && resource == scioxide_core::quantity!("sparse_entries"))
        );
    }
    let bounded = SparseProductPlan::with_options(
        &left,
        &right,
        SparseProductPlanOptions::new().with_max_output_entries(8),
    )
    .unwrap()
    .create_output::<f64>()
    .unwrap();
    let unbounded = SparseProductPlan::new(&left, &right)
        .unwrap()
        .create_output::<f64>()
        .unwrap();
    assert_eq!(bounded, unbounded);
    assert_eq!(bounded.col_indices(), &[0, 1, 2, 3, 0, 1, 2, 3]);
    let empty = CsrMatrix::from_raw(2, 4, vec![0; 3], vec![], vec![]).unwrap();
    assert_eq!(
        SparseProductPlan::with_options(
            &empty,
            &right,
            SparseProductPlanOptions::new().with_max_output_entries(0)
        )
        .unwrap()
        .create_output::<f64>()
        .unwrap()
        .nnz(),
        0
    );
    assert_eq!(
        SparseProductPlan::with_options(
            &right,
            &left,
            SparseProductPlanOptions::new().with_max_output_entries(0)
        )
        .unwrap_err()
        .kind(),
        scioxide_core::SciOxideErrorKind::DimensionMismatch
    );
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

// Row-major dense copy of the structural entries of a view.
fn dense_of<S: NumericScalar>(view: scioxide_sparse::SparseView<'_, S>) -> Vec<S> {
    let mut dense = vec![S::ZERO; view.rows() * view.cols()];
    for (r, c, v) in view.iter() {
        dense[r * view.cols() + c] = v;
    }
    dense
}
// Dense (conjugate) transpose of a row-major `rows x cols` array.
fn dense_transpose<S: NumericScalar>(
    dense: &[S],
    (rows, cols): (usize, usize),
    conjugate: bool,
) -> Vec<S> {
    (0..cols * rows)
        .map(|k| {
            let v = dense[(k % rows) * cols + k / rows];
            if conjugate { v.conj() } else { v }
        })
        .collect()
}
// The previous builder round trip: every stored slot, sorted and policy-resolved.
fn builder_csr<S: NumericScalar>(
    view: scioxide_sparse::SparseView<'_, S>,
    zeros: ExplicitZeroPolicy,
) -> CsrMatrix<S> {
    let mut builder = SparseBuilder::new(view.rows(), view.cols());
    view.iter()
        .try_for_each(|(r, c, v)| builder.push(r, c, v))
        .unwrap();
    builder.finish_csr(DuplicatePolicy::Sum, zeros).unwrap()
}
fn builder_transpose<S: NumericScalar>(
    view: scioxide_sparse::SparseView<'_, S>,
    conjugate: bool,
) -> scioxide_sparse::CooMatrix<S> {
    let mut builder = SparseBuilder::new(view.cols(), view.rows());
    view.iter()
        .try_for_each(|(r, c, v)| builder.push(c, r, if conjugate { v.conj() } else { v }))
        .unwrap();
    builder
        .finish_coo(DuplicatePolicy::Sum, ExplicitZeroPolicy::Preserve)
        .unwrap()
}
fn check_conversions<S: NumericScalar>(view: scioxide_sparse::SparseView<'_, S>) {
    for zeros in [ExplicitZeroPolicy::Preserve, ExplicitZeroPolicy::Drop] {
        assert_eq!(view.to_csr(zeros).unwrap(), builder_csr(view, zeros));
    }
    let preserved = view.to_csr(ExplicitZeroPolicy::Preserve).unwrap();
    assert_eq!(dense_of(preserved.view()), dense_of(view));
    assert_eq!(preserved.nnz(), view.iter().count());
}
fn transposed_formats<S: NumericScalar>() {
    let v = |re: f64, im: f64| scalar::<S>(re, im);
    let z = S::ZERO;
    // 4 x 6 in non-square 2 x 3 blocks: block (0, 0), an all-zero block (0, 1)
    // and block (1, 1) with zero slots.
    let bsr = BsrMatrix::from_raw(
        4,
        6,
        2,
        3,
        vec![0, 2, 3],
        vec![0, 1, 1],
        vec![
            v(1., 2.),
            v(2., -1.),
            z,
            v(3., 0.),
            v(4., 4.),
            v(5., -3.),
            z,
            z,
            z,
            z,
            z,
            z,
            v(6., 1.),
            z,
            v(7., -2.),
            z,
            v(8., 0.),
            z,
        ],
    )
    .unwrap();
    // 3 x 5 with diagonals -2, 0 and 3, including stored zeros.
    let dia = DiaMatrix::from_raw(
        3,
        5,
        vec![-2, 0, 3],
        vec![
            v(1., -1.),
            z,
            z,
            z,
            z,
            v(2., 1.),
            z,
            v(3., 3.),
            z,
            z,
            z,
            z,
            z,
            v(4., -4.),
            v(5., 2.),
        ],
    )
    .unwrap();
    let empty_bsr = BsrMatrix::<S>::from_raw(4, 6, 2, 3, vec![0, 0, 0], vec![], vec![]).unwrap();
    let flat_bsr = BsrMatrix::<S>::from_raw(0, 6, 2, 3, vec![0], vec![], vec![]).unwrap();
    let empty_dia = DiaMatrix::<S>::from_raw(3, 5, vec![], vec![]).unwrap();
    for matrix in [&bsr, &empty_bsr, &flat_bsr] {
        let (br, bc) = matrix.block_shape();
        let shape = (matrix.rows(), matrix.cols());
        check_conversions(matrix.view());
        for conjugate in [false, true] {
            let t = if conjugate {
                matrix.adjoint().unwrap()
            } else {
                matrix.transpose().unwrap()
            };
            assert_eq!(
                ((t.rows(), t.cols()), t.block_shape()),
                ((shape.1, shape.0), (bc, br))
            );
            let reference = builder_transpose(matrix.view(), conjugate);
            assert_eq!(t, BsrMatrix::from_sparse(reference.view(), bc, br).unwrap());
            assert_eq!(
                dense_of(t.view()),
                dense_transpose(&dense_of(matrix.view()), shape, conjugate)
            );
            check_conversions(t.view());
            let back = if conjugate {
                t.adjoint()
            } else {
                t.transpose()
            };
            assert_eq!(&back.unwrap(), matrix);
        }
    }
    for matrix in [&dia, &empty_dia] {
        let shape = (matrix.rows(), matrix.cols());
        check_conversions(matrix.view());
        for conjugate in [false, true] {
            let t = if conjugate {
                matrix.adjoint().unwrap()
            } else {
                matrix.transpose().unwrap()
            };
            assert_eq!((t.rows(), t.cols()), (shape.1, shape.0));
            let reference = builder_transpose(matrix.view(), conjugate);
            assert_eq!(t, DiaMatrix::from_sparse(reference.view()).unwrap());
            assert_eq!(
                dense_of(t.view()),
                dense_transpose(&dense_of(matrix.view()), shape, conjugate)
            );
            check_conversions(t.view());
            let back = if conjugate {
                t.adjoint()
            } else {
                t.transpose()
            };
            assert_eq!(&back.unwrap(), matrix);
        }
    }
    assert_eq!(dia.transpose().unwrap().offsets(), &[-3, 0, 2]);
    let csr = bsr.to_csr(ExplicitZeroPolicy::Drop).unwrap();
    for view in [
        csr.view(),
        csr.to_csc().unwrap().view(),
        csr.to_coo().unwrap().view(),
    ] {
        check_conversions(view);
    }
}
#[test]
fn bsr_and_dia_transposes_reuse_storage_order() {
    transposed_formats::<f64>();
    transposed_formats::<Complex<f64>>();
    transposed_formats::<Complex<f32>>();
}

// Owned products equal the `_into` forms bit for bit in every storage format.
fn owned_products<S: NumericScalar>() {
    use scioxide_core::SciOxideErrorKind;
    let entries = [
        (0, 0, (1.5, -0.5)),
        (0, 3, (2.0, 1.0)),
        (1, 1, (-3.0, 0.25)),
        (2, 0, (0.0, 0.0)),
    ];
    let (rows, cols): (Vec<_>, Vec<_>) = entries.iter().map(|&(i, j, _)| (i, j)).unzip();
    let values: Vec<S> = entries
        .iter()
        .map(|&(_, _, (re, im))| scalar(re, im))
        .collect();
    let csr = CsrMatrix::from_triplets(3, 4, &rows, &cols, &values).unwrap();
    let csc = csr.to_csc().unwrap();
    let coo = csr.to_coo().unwrap();
    let bsr = BsrMatrix::from_sparse(csr.view(), 1, 2).unwrap();
    let dia = DiaMatrix::from_sparse(csr.view()).unwrap();
    let x: Vec<S> = (0..4).map(|i| scalar(i as f64 + 1.0, 0.5)).collect();
    let y: Vec<S> = (0..3).map(|i| scalar(0.5 - i as f64, 1.0)).collect();
    let into = |view: scioxide_sparse::SparseView<'_, S>, operation| {
        let (input, len) = match operation {
            SparseOperation::Normal => (&x, 3),
            _ => (&y, 4),
        };
        let mut output = vec![S::ZERO; len];
        let mut scratch = vec![S::ZERO; len];
        match operation {
            SparseOperation::Normal => view.apply_into(input, &mut output, &mut scratch),
            SparseOperation::Transpose => {
                view.apply_transpose_into(input, &mut output, &mut scratch)
            }
            SparseOperation::Adjoint => view.apply_adjoint_into(input, &mut output, &mut scratch),
        }
        .unwrap();
        output
    };
    for view in [csr.view(), csc.view(), coo.view(), bsr.view(), dia.view()] {
        assert_eq!(view.apply(&x).unwrap(), into(view, SparseOperation::Normal));
        assert_eq!(
            view.apply_transpose(&y).unwrap(),
            into(view, SparseOperation::Transpose)
        );
        assert_eq!(
            view.apply_adjoint(&y).unwrap(),
            into(view, SparseOperation::Adjoint)
        );
        let exact = view.with_accuracy(scioxide_core::Accuracy::Exact);
        assert_eq!(
            exact.apply(&x).unwrap(),
            into(view, SparseOperation::Normal)
        );
        assert_eq!(
            view.apply(&x[..3]).unwrap_err().kind(),
            SciOxideErrorKind::DimensionMismatch
        );
        let mut nan = x.clone();
        nan[1] = S::from_real(S::Real::NAN);
        assert_eq!(
            view.apply(&nan).unwrap_err().kind(),
            SciOxideErrorKind::NonFinite
        );
    }
    let normal = into(csr.view(), SparseOperation::Normal);
    let adjoint = into(csr.view(), SparseOperation::Adjoint);
    assert_eq!(
        (csr.apply(&x).unwrap(), csr.apply_adjoint(&y).unwrap()),
        (normal.clone(), adjoint.clone())
    );
    assert_eq!(
        (csc.apply(&x).unwrap(), csc.apply_adjoint(&y).unwrap()),
        (normal.clone(), adjoint.clone())
    );
    assert_eq!(
        (bsr.apply(&x).unwrap(), bsr.apply_adjoint(&y).unwrap()),
        (normal.clone(), adjoint.clone())
    );
    assert_eq!(
        (dia.apply(&x).unwrap(), dia.apply_adjoint(&y).unwrap()),
        (normal, adjoint)
    );
}
#[test]
fn owned_products_equal_into_forms() {
    owned_products::<f32>();
    owned_products::<f64>();
    owned_products::<scioxide_core::Complex<f32>>();
    owned_products::<scioxide_core::Complex<f64>>();
}
#[test]
fn from_triplets_sums_duplicates_and_keeps_every_coordinate() {
    use scioxide_core::SciOxideErrorKind;
    let (rows, cols, values) = (
        [1, 0, 1, 0, 2],
        [2, 0, 2, 1, 1],
        [2.0_f64, 1.0, 3.0, 4.0, 0.0],
    );
    let a = CsrMatrix::from_triplets(3, 3, &rows, &cols, &values).unwrap();
    let mut builder = SparseBuilder::new(3, 3);
    builder.extend(&rows, &cols, &values).unwrap();
    assert_eq!(
        a,
        builder
            .finish_csr(DuplicatePolicy::Sum, ExplicitZeroPolicy::Preserve)
            .unwrap()
    );
    // Reversed triplets give the same matrix, and the stored zero stays.
    let reversed = |v: &[usize]| v.iter().rev().copied().collect::<Vec<_>>();
    let values_reversed: Vec<_> = values.iter().rev().copied().collect();
    let b = CsrMatrix::from_triplets(3, 3, &reversed(&rows), &reversed(&cols), &values_reversed)
        .unwrap();
    assert_eq!(a, b);
    assert_eq!(
        a.iter().collect::<Vec<_>>(),
        [(0, 0, 1.0), (0, 1, 4.0), (1, 2, 5.0), (2, 1, 0.0)]
    );
    // A cancelling duplicate stays as a stored zero.
    let cancel = CsrMatrix::from_triplets(1, 1, &[0, 0], &[0, 0], &[1.5_f64, -1.5]).unwrap();
    assert_eq!((cancel.nnz(), cancel.values()), (1, &[0.0][..]));
    let kind = |r: Result<CsrMatrix<f64>, scioxide_core::SciOxideError>| r.unwrap_err().kind();
    assert_eq!(
        kind(CsrMatrix::from_triplets(2, 2, &[0], &[0, 1], &[1.0])),
        SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(
        kind(CsrMatrix::from_triplets(2, 2, &[2], &[0], &[1.0])),
        SciOxideErrorKind::OutOfBounds
    );
    assert_eq!(
        kind(CsrMatrix::from_triplets(2, 2, &[0], &[0], &[f64::NAN])),
        SciOxideErrorKind::NonFinite
    );
    let overflow = CsrMatrix::from_triplets(1, 1, &[0, 0], &[0, 0], &[f64::MAX, f64::MAX]);
    assert_eq!(kind(overflow), SciOxideErrorKind::NumericalInstability);
}
#[test]
fn identity_stores_the_unit_diagonal() {
    let eye = CsrMatrix::<scioxide_core::Complex<f64>>::identity(3).unwrap();
    assert_eq!((eye.rows(), eye.cols(), eye.nnz()), (3, 3, 3));
    assert!(
        eye.iter()
            .all(|(i, j, v)| i == j && v == scioxide_core::Complex::new(1.0, 0.0))
    );
    let empty = CsrMatrix::<f64>::identity(0).unwrap();
    assert_eq!((empty.rows(), empty.nnz()), (0, 0));
}
#[test]
fn arithmetic_one_shots_equal_their_configured_forms() {
    use scioxide_sparse::SparseArithmeticOptions;
    let defaults = SparseArithmeticOptions::default();
    assert_eq!(defaults, SparseArithmeticOptions::new());
    assert_eq!(defaults.zero_policy(), ExplicitZeroPolicy::Preserve);
    assert_eq!(defaults.accuracy(), scioxide_core::Accuracy::Standard);
    let a = CsrMatrix::from_triplets(2, 3, &[0, 0, 1], &[0, 1, 2], &[1.0_f64, 2.0, 3.0]).unwrap();
    let b = CsrMatrix::from_triplets(2, 3, &[0, 1, 1], &[1, 0, 2], &[-2.0, 4.0, 1.0]).unwrap();
    assert_eq!(
        a.add(&b).unwrap(),
        a.add_with_options(&b, defaults).unwrap()
    );
    assert_eq!(
        a.subtract(&b).unwrap(),
        a.subtract_with_options(&b, defaults).unwrap()
    );
    assert_eq!(
        a.hadamard(&b).unwrap(),
        a.hadamard_with_options(&b, defaults).unwrap()
    );
    assert_eq!(
        a.view().scaled(0.0).unwrap(),
        a.view().scaled_with_options(0.0, defaults).unwrap()
    );
    // Preserve keeps the cancelled coordinate (0, 1) and the zero product.
    assert_eq!(a.add(&b).unwrap().get(0, 1).unwrap(), 0.0);
    assert_eq!(a.add(&b).unwrap().nnz(), 4);
    assert_eq!(a.view().scaled(0.0).unwrap().nnz(), 3);
    let product = a
        .multiply(&b.transpose().unwrap().to_csr().unwrap())
        .unwrap();
    let plan = SparseProductPlan::new(&a, &b.transpose().unwrap().to_csr().unwrap()).unwrap();
    assert_eq!(product.row_ptr(), plan.row_ptr());
    assert_eq!(product.col_indices(), plan.col_indices());
}
#[test]
fn sparse_diagonals_sum_repeated_offsets_and_keep_zeros() {
    let a = sparse_diagonals(
        3,
        3,
        &[0, 1, 0],
        &[&[1.0_f64, 0.0, 2.0], &[5.0, 6.0], &[1.0, 1.0, 1.0]],
    )
    .unwrap();
    assert_eq!(
        a.iter()
            .filter(|&(i, j, _)| i == j)
            .map(|(_, _, v)| v)
            .collect::<Vec<_>>(),
        [2.0, 1.0, 3.0]
    );
    assert_eq!(a.nnz(), 5);
    let zero = sparse_diagonals(2, 2, &[0], &[&[0.0_f64, 1.0]]).unwrap();
    assert_eq!((zero.nnz(), zero.get(0, 0).unwrap()), (2, 0.0));
}
#[test]
fn product_plan_options_bound_only_the_output_count() {
    use scioxide_core::SciOxideErrorKind;
    // [[1, 2], [0, 3]]² has three structural entries.
    let a = CsrMatrix::from_triplets(2, 2, &[0, 0, 1], &[0, 1, 1], &[1.0_f64, 2.0, 3.0]).unwrap();
    let unbounded = SparseProductPlanOptions::new();
    assert_eq!(
        (
            unbounded,
            SparseProductPlanOptions::default().max_output_entries()
        ),
        (SparseProductPlanOptions::default(), usize::MAX)
    );
    let plan = SparseProductPlan::new(&a, &a).unwrap();
    let configured = SparseProductPlan::with_options(&a, &a, unbounded).unwrap();
    assert_eq!(
        (plan.row_ptr(), plan.col_indices()),
        (configured.row_ptr(), configured.col_indices())
    );
    let error =
        SparseProductPlan::with_options(&a, &a, unbounded.with_max_output_entries(2)).unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::ResourceLimit);
    assert!(SparseProductPlan::with_options(&a, &a, unbounded.with_max_output_entries(3)).is_ok());
}
