//! Representation-independent sparse construction and kernel contracts.

#![allow(clippy::unwrap_used)]

use scioxide_core::{BlockOperator, BlockVectorView, BlockVectorViewMut, Preconditioner};
use scioxide_sparse::{DuplicatePolicy, ExplicitZeroPolicy, JacobiPreconditioner, SparseBuilder};

fn shuffled_indices(length: usize, mut state: u64) -> Vec<usize> {
    let mut indices = (0..length).collect::<Vec<_>>();
    (1..length).rev().for_each(|upper| {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        indices.swap(upper, (state as usize) % (upper + 1));
    });
    indices
}

#[test]
fn coo_permutations_and_duplicates_round_trip_through_csr_and_csc() {
    let rows = [2, 0, 1, 2, 0, 1, 2, 1, 0, 2];
    let cols = [0, 3, 2, 0, 1, 2, 3, 1, 3, 3];
    let values = [4.0_f64, 1.0, 5.0, -1.0, -2.0, -3.0, 7.0, 6.0, 2.0, -7.0];

    let reference = {
        let mut builder = scioxide_sparse::SparseBuilder::new(3, 4);
        builder
            .extend(&rows, &cols, &values)
            .and_then(|()| builder.finish_coo(DuplicatePolicy::Sum, ExplicitZeroPolicy::Drop))
    }
    .unwrap();
    let mut expected_dense = vec![0.0; 12];
    reference
        .iter()
        .for_each(|(row, col, value)| expected_dense[row * 4 + col] = value);

    (0..64_u64).for_each(|seed| {
        let order = shuffled_indices(values.len(), seed);
        let mut builder = SparseBuilder::new(3, 4);
        order
            .iter()
            .try_for_each(|&index| builder.push(rows[index], cols[index], values[index]))
            .unwrap();
        let direct_csr = builder
            .clone()
            .finish_csr(DuplicatePolicy::Sum, ExplicitZeroPolicy::Drop)
            .unwrap();
        let direct_csc = builder
            .clone()
            .finish_csc(DuplicatePolicy::Sum, ExplicitZeroPolicy::Drop)
            .unwrap();
        let coo = builder
            .finish_coo(DuplicatePolicy::Sum, ExplicitZeroPolicy::Drop)
            .unwrap();
        assert_eq!(coo, reference);
        assert_eq!(direct_csr, reference.to_csr().unwrap());
        assert_eq!(direct_csc, reference.to_csc().unwrap());
        assert_eq!(
            scioxide_sparse::BsrMatrix::from_sparse(coo.view(), 1, 2).unwrap(),
            scioxide_sparse::BsrMatrix::from_sparse(reference.view(), 1, 2).unwrap()
        );
        assert_eq!(
            scioxide_sparse::DiaMatrix::from_sparse(coo.view()).unwrap(),
            scioxide_sparse::DiaMatrix::from_sparse(reference.view()).unwrap()
        );

        let csr = coo.to_csr().unwrap();
        let csc = coo.to_csc().unwrap();
        assert_eq!(csr.to_csc().unwrap().to_csr().unwrap(), csr);
        assert_eq!(csc.to_csr().unwrap().to_csc().unwrap(), csc);
        assert_eq!(csr.to_coo().unwrap(), reference);
        assert_eq!(csc.to_coo().unwrap(), reference);

        let mut csr_dense = vec![f64::NAN; 12];
        let mut csc_dense = vec![f64::NAN; 12];
        csr.to_dense_into(&mut csr_dense).unwrap();
        csc.to_dense_into(&mut csc_dense).unwrap();
        assert_eq!(csr_dense, expected_dense);
        assert_eq!(csc_dense, expected_dense);
    });
}

#[test]
fn builder_is_transactional_and_reports_allocation_failure() {
    let mut csr = SparseBuilder::new(2, 2);
    csr.extend(&[0], &[0], &[1.0_f64]).unwrap();
    let before = csr.triplet_count();
    assert!(csr.extend(&[1, 2], &[1, 0], &[2.0, 3.0]).is_err());
    assert_eq!(csr.triplet_count(), before);

    let mut csc = SparseBuilder::new(2, 2);
    csc.extend(&[0], &[1], &[2.0_f64]).unwrap();
    let before = csc.triplet_count();
    assert!(csc.extend(&[0], &[0, 1], &[1.0, 2.0]).is_err());
    assert_eq!(csc.triplet_count(), before);

    assert!(SparseBuilder::<f64>::with_capacity(1, 1, usize::MAX).is_err());
}

#[test]
fn extreme_scale_matvec_and_block_matmat_match_dense_reference() {
    let matrix = {
        let mut builder = scioxide_sparse::SparseBuilder::new(2, 3);
        builder
            .extend(
                &[0, 0, 0, 1, 1],
                &[0, 1, 2, 0, 1],
                &[1.0e300_f64, -1.0e300, 3.0, 1.0e-300, 2.0e-300],
            )
            .and_then(|()| builder.finish_csr(DuplicatePolicy::Sum, ExplicitZeroPolicy::Drop))
    }
    .unwrap();
    let input_values = [1.0, 1.0, 1.0, 2.0, 2.0, -1.0];
    let mut output_values = [9.0; 4];
    let input = BlockVectorView::new(&input_values, 3, 2).unwrap();
    let mut output = BlockVectorViewMut::new(&mut output_values, 2, 2).unwrap();
    let workspace_len = matrix.apply_block_workspace_len(2).unwrap();
    let mut workspace = vec![0.0; workspace_len];
    matrix
        .apply_block_into(input, &mut output, &mut workspace)
        .unwrap();

    assert_eq!(output_values[0], 3.0);
    assert_eq!(output_values[1], 3.0e-300);
    assert_eq!(output_values[2], -3.0);
    assert_eq!(output_values[3], 6.0e-300);

    let mut unchanged = [7.0; 4];
    let input = BlockVectorView::new(&input_values, 3, 2).unwrap();
    let mut output = BlockVectorViewMut::new(&mut unchanged, 2, 2).unwrap();
    assert!(
        matrix
            .apply_block_into(input, &mut output, &mut workspace[..workspace_len - 1])
            .is_err()
    );
    assert_eq!(unchanged, [7.0; 4]);
}

#[test]
fn block_preconditioning_stages_every_right_hand_side() {
    let matrix = {
        let mut builder = scioxide_sparse::SparseBuilder::new(2, 2);
        builder
            .extend(&[0, 1], &[0, 1], &[2.0_f64, 4.0])
            .and_then(|()| builder.finish_csr(DuplicatePolicy::Sum, ExplicitZeroPolicy::Drop))
    }
    .unwrap();
    let jacobi = JacobiPreconditioner::new(&matrix).unwrap();
    let input_values = [2.0, 8.0, 6.0, 4.0];
    let mut output_values = [7.0; 4];
    let input = BlockVectorView::new(&input_values, 2, 2).unwrap();
    let mut output = BlockVectorViewMut::new(&mut output_values, 2, 2).unwrap();
    let required = jacobi.precondition_block_workspace_len(2).unwrap();
    let mut workspace = vec![0.0; required];
    jacobi
        .precondition_block_into(input, &mut output, &mut workspace)
        .unwrap();
    assert_eq!(output_values, [1.0, 2.0, 3.0, 1.0]);

    let input = BlockVectorView::new(&input_values, 2, 2).unwrap();
    let mut output = BlockVectorViewMut::new(&mut output_values, 2, 2).unwrap();
    assert!(
        jacobi
            .precondition_block_into(input, &mut output, &mut workspace[..required - 1])
            .is_err()
    );
    assert_eq!(output_values, [1.0, 2.0, 3.0, 1.0]);
}

#[test]
fn duplicate_cancellation_is_range_safe_for_both_real_precisions() {
    let single_f32 = {
        let mut builder = scioxide_sparse::SparseBuilder::new(1, 1);
        builder
            .extend(&[0, 0, 0], &[0, 0, 0], &[1.0e30_f32, -1.0e30, 1.0])
            .and_then(|()| builder.finish_coo(DuplicatePolicy::Sum, ExplicitZeroPolicy::Drop))
    }
    .unwrap();
    let single_f64 = {
        let mut builder = scioxide_sparse::SparseBuilder::new(1, 1);
        builder
            .extend(&[0, 0, 0], &[0, 0, 0], &[1.0e300_f64, -1.0e300, 1.0])
            .and_then(|()| builder.finish_coo(DuplicatePolicy::Sum, ExplicitZeroPolicy::Drop))
    }
    .unwrap();
    assert_eq!(single_f32.values(), &[1.0]);
    assert_eq!(single_f64.values(), &[1.0]);
}

fn construction<S: scioxide_core::NumericScalar>() -> SparseBuilder<S> {
    let mut b = SparseBuilder::new(4, 6);
    let two = S::ONE + S::ONE;
    b.extend(
        &[3, 0, 2, 0, 3, 1],
        &[4, 1, 2, 1, 4, 5],
        &[two, S::ONE, S::ZERO, two, -two, two],
    )
    .unwrap();
    // Include a non-real entry when the scalar supports complex arithmetic.
    use scioxide_core::AlgebraicScalar;
    let imaginary = if !S::IS_REAL {
        S::from_components(S::Real::ZERO, S::Real::ONE)
    } else {
        S::ONE
    };
    b.push(2, 0, imaginary).unwrap();
    b
}

fn construction_contract<S: scioxide_core::NumericScalar>() {
    use scioxide_sparse::{BsrMatrix, DiaMatrix};
    for zeros in [ExplicitZeroPolicy::Preserve, ExplicitZeroPolicy::Drop] {
        let builder = construction::<S>();
        let coo = builder
            .clone()
            .finish_coo(DuplicatePolicy::Sum, zeros)
            .unwrap();
        let csr = builder
            .clone()
            .finish_csr(DuplicatePolicy::Sum, zeros)
            .unwrap();
        let csc = builder.finish_csc(DuplicatePolicy::Sum, zeros).unwrap();
        assert_eq!(coo.to_csr().unwrap(), csr);
        assert_eq!(coo.to_csc().unwrap(), csc);
        assert_eq!(csr.structural_pattern(), csc.structural_pattern());
        assert_eq!(csr.to_csc().unwrap().to_csr().unwrap(), csr);
        assert_eq!(csc.to_csr().unwrap().to_csc().unwrap(), csc);
        assert_eq!(csr.transpose().unwrap().transpose().unwrap(), csr);
        assert_eq!(csc.adjoint().unwrap().adjoint().unwrap(), csc);
        assert_eq!(
            csr.nnz(),
            if zeros == ExplicitZeroPolicy::Drop {
                3
            } else {
                5
            }
        );
        let bsr = BsrMatrix::from_sparse(coo.view(), 2, 3).unwrap();
        let dia = DiaMatrix::from_sparse(coo.view()).unwrap();
        let mut expected = [S::ZERO; 24];
        csr.to_dense_into(&mut expected).unwrap();
        for converted in [bsr.to_csr(zeros).unwrap(), dia.to_csr(zeros).unwrap()] {
            let mut actual = [S::ZERO; 24];
            converted.to_dense_into(&mut actual).unwrap();
            assert_eq!(actual, expected);
        }
        // Rejected duplicates cannot reach any downstream format conversion.
        let b = construction::<S>();
        assert!(
            b.clone()
                .finish_coo(DuplicatePolicy::Reject, zeros)
                .is_err()
        );
        assert!(
            b.clone()
                .finish_csr(DuplicatePolicy::Reject, zeros)
                .is_err()
        );
        assert!(b.finish_csc(DuplicatePolicy::Reject, zeros).is_err());
    }
}

#[test]
fn direct_construction_covers_all_formats_and_scalar_families() {
    construction_contract::<f32>();
    construction_contract::<f64>();
    construction_contract::<scioxide_core::Complex<f32>>();
    construction_contract::<scioxide_core::Complex<f64>>();
}

fn traversal_contract<S: scioxide_core::NumericScalar>() {
    use scioxide_core::AlgebraicScalar;
    let mut builder = SparseBuilder::new(4, 6);
    let value = S::from_components(
        S::Real::ONE,
        if !S::IS_REAL {
            S::Real::ONE
        } else {
            S::Real::ZERO
        },
    );
    builder
        .extend(
            &[0, 0, 2, 3],
            &[1, 5, 1, 0],
            &[value, -value, S::ZERO, S::ONE],
        )
        .unwrap();
    let mut csr = builder
        .clone()
        .finish_csr(DuplicatePolicy::Reject, ExplicitZeroPolicy::Preserve)
        .unwrap();
    let mut csc = builder
        .finish_csc(DuplicatePolicy::Reject, ExplicitZeroPolicy::Preserve)
        .unwrap();
    let expected_transpose = scioxide_sparse::transpose_pattern(csr.structural_pattern()).unwrap();
    assert_eq!(csr.opposite_traversal(), expected_transpose.view());
    assert_eq!(csc.opposite_traversal().pattern(), csr.structural_pattern());
    for (traversal, values, is_csr) in [
        (csr.opposite_traversal(), csr.values(), true),
        (csc.opposite_traversal(), csc.values(), false),
    ] {
        let pattern = traversal.pattern();
        let mut seen = [false; 4];
        for (major, bounds) in pattern.row_ptr().windows(2).enumerate() {
            for k in bounds[0]..bounds[1] {
                let p = traversal.positions()[k];
                assert!(!seen[p]);
                seen[p] = true;
                let minor = pattern.col_indices()[k];
                let (row, col) = if is_csr {
                    (minor, major)
                } else {
                    (major, minor)
                };
                assert_eq!(values[p], csr.get(row, col).unwrap());
                assert_eq!(values[p], csc.get(row, col).unwrap());
            }
        }
        assert!(seen.into_iter().all(|v| v));
    }
    for _ in 0..4 {
        assert_eq!(csr.structural_pattern(), csc.structural_pattern());
        assert_eq!(csr.opposite_traversal(), expected_transpose.view());
        assert_eq!(csc.opposite_traversal().positions().len(), 4);
    }
    let before = csr.structural_pattern().to_owned().unwrap();
    csr.set_value(0, -value).unwrap();
    csc.set_value(0, -value).unwrap();
    assert_eq!(csr.structural_pattern(), before.view());
    assert_eq!(csc.structural_pattern(), before.view());
    let adjoint = csr.adjoint().unwrap();
    for (r, c, v) in csr.iter() {
        assert_eq!(adjoint.get(c, r).unwrap(), v.conj());
    }
}

#[test]
fn cached_traversals_preserve_positions_for_real_and_complex_values() {
    traversal_contract::<f32>();
    traversal_contract::<f64>();
    traversal_contract::<scioxide_core::Complex<f32>>();
    traversal_contract::<scioxide_core::Complex<f64>>();
}

#[test]
fn empty_patterns_and_raw_validation_share_core_invariants() {
    use scioxide_sparse::{CscMatrix, CsrMatrix};
    for (rows, cols) in [(0, 0), (0, 3), (4, 0), (4, 6)] {
        let b = SparseBuilder::<f64>::new(rows, cols);
        let csr = b
            .clone()
            .finish_csr(DuplicatePolicy::Sum, ExplicitZeroPolicy::Drop)
            .unwrap();
        let csc = b
            .finish_csc(DuplicatePolicy::Sum, ExplicitZeroPolicy::Drop)
            .unwrap();
        assert_eq!(csr.structural_pattern(), csc.structural_pattern());
        assert_eq!(csr.structural_pattern().row_ptr().len(), rows + 1);
        assert_eq!(csr.opposite_traversal().pattern().row_ptr().len(), cols + 1);
        assert_eq!(csr.to_csc().unwrap(), csc);
        assert_eq!(csc.to_csr().unwrap(), csr);
    }
    for (pointers, indices) in [
        (vec![1, 1, 1], vec![]),
        (vec![0, 2, 1], vec![0]),
        (vec![0, 1, 1], vec![2]),
        (vec![0, 2, 2], vec![1, 0]),
        (vec![0, 2, 2], vec![0, 0]),
    ] {
        let values = vec![1.0; indices.len()];
        assert!(
            CsrMatrix::from_raw(2, 2, pointers.clone(), indices.clone(), values.clone()).is_err()
        );
        assert!(CscMatrix::from_raw(2, 2, pointers, indices, values).is_err());
    }
    assert!(CsrMatrix::<f64>::from_raw(1, 1, vec![0, 1], vec![0], vec![]).is_err());
    assert!(CscMatrix::<f64>::from_raw(1, 1, vec![0, 1], vec![0], vec![f64::NAN]).is_err());
    let huge = SparseBuilder::<f64>::new(usize::MAX, 0);
    assert!(
        huge.clone()
            .finish_csr(DuplicatePolicy::Sum, ExplicitZeroPolicy::Drop)
            .is_err()
    );
    assert!(
        huge.finish_csc(DuplicatePolicy::Sum, ExplicitZeroPolicy::Drop)
            .is_err()
    );
}

// Owned `apply` equals `apply_into` with the default options for each
// preconditioner, and the defaults of the preconditioner options.
#[test]
fn owned_preconditioner_applications_equal_their_into_forms() {
    use scioxide_core::{ExecutionContext, SciOxideErrorKind, Tolerance};
    use scioxide_sparse::{
        AmgCycleOptions, AmgLimits, AmgPreconditioner, IdentityPreconditioner, SchwarzKind,
        SchwarzLocalSolveView, SchwarzOptions, SchwarzPlan, SchwarzSubdomainView,
        SparseSolveOptions, sparse_diagonals,
    };
    let a = sparse_diagonals(5, 5, &[-1, 0, 1], &[&[-1.0_f64; 4], &[2.0; 5], &[-1.0; 4]]).unwrap();
    let rhs = [1.0, -0.5, 0.25, 2.0, 0.0];
    macro_rules! check {
        ($preconditioner:expr) => {{
            let p = &$preconditioner;
            let mut output = [0.0; 5];
            p.apply_into(
                &rhs,
                SparseSolveOptions::new(),
                &mut output,
                &mut p.workspace(1).unwrap(),
                ExecutionContext::sequential(),
            )
            .unwrap();
            assert_eq!(p.apply(&rhs).unwrap(), output);
        }};
    }
    check!(IdentityPreconditioner::<f64>::new(5));
    let jacobi = JacobiPreconditioner::new(&a).unwrap();
    check!(jacobi);
    let amg = AmgPreconditioner::new(&a).unwrap();
    check!(amg);
    let coarse = scioxide_sparse::SparseCholesky::new(amg.hierarchy().coarse_matrix()).unwrap();
    check!(
        amg.hierarchy()
            .cycle(&coarse, AmgCycleOptions::new())
            .unwrap()
    );
    let domain = |indices, weights| SchwarzSubdomainView {
        indices,
        weights,
        solve: SchwarzLocalSolveView::Preconditioner(&jacobi),
    };
    let unit = [1.0; 5];
    let domains = [domain(&[0, 1, 2, 3, 4][..], Some(&unit[..]))];
    let plan = SchwarzPlan::new(5, SchwarzKind::Additive, &domains, None).unwrap();
    check!(plan);
    let options = SchwarzOptions::default();
    assert_eq!(options, SchwarzOptions::new());
    assert_eq!(
        options.partition_tolerance(),
        Tolerance::from_machine_epsilon()
    );
    assert_eq!(
        (options.max_plan_bytes(), options.max_workspace_bytes()),
        (usize::MAX, usize::MAX)
    );
    let configured =
        SchwarzPlan::with_options(5, SchwarzKind::Additive, &domains, None, options).unwrap();
    assert_eq!(configured.plan_bytes(), plan.plan_bytes());
    assert_eq!(configured.apply(&rhs).unwrap(), plan.apply(&rhs).unwrap());
    let error = SchwarzPlan::with_options(
        5,
        SchwarzKind::Additive,
        &domains,
        None,
        options.with_max_plan_bytes(0),
    )
    .err()
    .map(|error| error.kind());
    assert_eq!(error, Some(SciOxideErrorKind::ResourceLimit));
    let limits = AmgLimits::default();
    assert_eq!(
        (
            limits.max_levels(),
            limits.max_entries(),
            limits.max_setup_bytes()
        ),
        (64, usize::MAX, usize::MAX)
    );
}
