#![allow(clippy::unwrap_used)]
use scioxide_core::{
    Accuracy, Complex, ExecutionContext, NumericScalar, Preconditioner, RealScalar,
};
use scioxide_sparse::*;
fn options<R: RealScalar>() -> IncompleteOptions<R> {
    IncompleteOptions::new()
}
fn limits() -> IncompleteLimits {
    IncompleteLimits::new(128, 1_000_000)
}
fn scalar<S: NumericScalar>(r: f64, i: f64) -> S {
    S::from_components(
        S::Real::try_from_f64_lossy(r).unwrap(),
        S::Real::try_from_f64_lossy(i).unwrap(),
    )
}
fn matrix<S: NumericScalar>(complex: bool) -> CsrMatrix<S> {
    let im = if complex { 0.25 } else { 0.0 };
    CsrMatrix::from_raw(
        4,
        4,
        vec![0, 3, 6, 9, 12],
        vec![0, 1, 3, 0, 1, 2, 1, 2, 3, 0, 2, 3],
        vec![
            scalar(6., 0.),
            scalar(1., im),
            scalar(-1., 0.),
            scalar(1., -im),
            scalar(5., 0.),
            scalar(1., im),
            scalar(1., -im),
            scalar(4., 0.),
            scalar(1., 0.),
            scalar(-1., 0.),
            scalar(1., 0.),
            scalar(3., 0.),
        ],
    )
    .unwrap()
}

fn saved_factors<S: NumericScalar>(domain: &str, complex: bool, accuracy: Accuracy) {
    let execution = SparseSolveOptions::new().with_accuracy(accuracy);
    let a = matrix::<S>(complex);
    let methods = [
        IncompleteMethod::Ilu { level: 0 },
        IncompleteMethod::Ilu { level: 1 },
        IncompleteMethod::Ilu { level: 3 },
        IncompleteMethod::Ilut,
        IncompleteMethod::Ic0,
        IncompleteMethod::Ict,
    ];
    let mut cases = 0;
    for line in include_str!("data/incomplete_factors.txt")
        .lines()
        .filter(|line| line.starts_with(domain))
    {
        let mut fields = line.split_whitespace();
        assert_eq!(fields.next(), Some(domain));
        let method = methods[fields.next().unwrap().parse::<usize>().unwrap()];
        let b = fields.next().unwrap().parse::<usize>().unwrap();
        let drop = fields.next().unwrap().parse::<f64>().unwrap();
        let cap = fields.next().unwrap().parse::<usize>().unwrap();
        let diagnostics: [usize; 3] =
            core::array::from_fn(|_| fields.next().unwrap().parse().unwrap());
        let p = IncompleteSymbolic::analyze(a.structural_pattern(), method, b, limits()).unwrap();
        let mut o = options();
        o = o.with_drop_tolerance(S::Real::try_from_f64_lossy(drop).unwrap());
        o = o.with_max_off_diagonal(cap);
        let f = IncompleteFactor::factor_scoped(
            p.clone(),
            a.view(),
            o,
            execution,
            ExecutionContext::sequential(),
        )
        .unwrap();
        let d = f.diagnostics();
        assert_eq!(
            [d.factor_blocks(), d.dropped_blocks(), d.repaired_pivots()],
            diagnostics,
            "{domain} {method:?} {b}"
        );
        let tolerance = S::Real::EPSILON.try_to_f64_exact().unwrap() * 8.;
        for (r, c, values) in f.blocks() {
            assert_eq!(r, fields.next().unwrap().parse::<usize>().unwrap());
            assert_eq!(c, fields.next().unwrap().parse::<usize>().unwrap());
            for value in values {
                for component in [value.real_component(), value.imaginary_component()] {
                    let expected = fields.next().unwrap().parse::<f64>().unwrap();
                    let actual = component.try_to_f64_exact().unwrap();
                    assert!(
                        (actual - expected).abs() <= tolerance * (1. + expected.abs()),
                        "{domain} {method:?} block {b}, ({r}, {c}): {actual} != {expected}"
                    );
                }
            }
        }
        for r in 0..4 / b {
            for &pivot in f.diagonal_pivots(r).unwrap() {
                assert_eq!(pivot, fields.next().unwrap().parse::<usize>().unwrap());
            }
        }
        assert_eq!(fields.next(), None);
        cases += 1;
    }
    assert_eq!(cases, 30);
}

#[test]
fn factors_match_pre_sparse_accumulator_fixtures() {
    for accuracy in [Accuracy::Standard, Accuracy::Exact] {
        saved_factors::<f32>("f32", false, accuracy);
        saved_factors::<f64>("f64", false, accuracy);
        saved_factors::<Complex<f32>>("c32", true, accuracy);
        saved_factors::<Complex<f64>>("c64", true, accuracy);
    }
}

#[test]
fn threshold_equality_and_predivision_ilut_dropping() {
    let a = CsrMatrix::from_raw(2, 2, vec![0, 1, 3], vec![0, 0, 1], vec![100., 1., 4.]).unwrap();
    let p =
        IncompleteSymbolic::analyze(a.structural_pattern(), IncompleteMethod::Ilut, 1, limits())
            .unwrap();
    let mut o = options();
    o = o.with_drop_tolerance(0.2);
    let f = IncompleteFactor::factor(p, a.view(), o).unwrap();
    let (_, _, lower) = f.blocks().find(|&(r, c, _)| r == 1 && c == 0).unwrap();
    assert!((lower[0] - 0.01_f64).abs() < 1e-15);
    assert_eq!(f.diagnostics().dropped_blocks(), 0);

    let a = CsrMatrix::from_raw(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![2., 0.5, 0.5, 2.],
    )
    .unwrap();
    for method in [IncompleteMethod::Ilut, IncompleteMethod::Ict] {
        let p = IncompleteSymbolic::analyze(a.structural_pattern(), method, 1, limits()).unwrap();
        o = o.with_drop_tolerance(0.25);
        let f = IncompleteFactor::factor(p.clone(), a.view(), o).unwrap();
        assert!(f.blocks().any(|(r, c, _)| r == 1 && c == 0));
        assert_eq!(f.diagnostics().dropped_blocks(), 0);
        o = o.with_drop_tolerance(0.25 + f64::EPSILON);
        let f = IncompleteFactor::factor(p.clone(), a.view(), o).unwrap();
        assert!(!f.blocks().any(|(r, c, _)| r == 1 && c == 0));
        assert_eq!(
            f.diagnostics().dropped_blocks(),
            if method == IncompleteMethod::Ilut {
                2
            } else {
                1
            }
        );
    }
}

#[test]
fn lower_retention_ties_and_diagonal_updates_follow_each_method() {
    let a = CsrMatrix::from_raw(
        3,
        3,
        vec![0, 2, 4, 7],
        vec![0, 2, 1, 2, 0, 1, 2],
        vec![4.0_f64, 1., 4., 1., 1., 1., 4.],
    )
    .unwrap();
    for method in [IncompleteMethod::Ilut, IncompleteMethod::Ict] {
        let p = IncompleteSymbolic::analyze(a.structural_pattern(), method, 1, limits()).unwrap();
        let mut o = options();
        o = o.with_max_off_diagonal(1);
        let f = IncompleteFactor::factor(p.clone(), a.view(), o).unwrap();
        assert!(f.blocks().any(|(r, c, _)| (r, c) == (2, 0)));
        assert!(!f.blocks().any(|(r, c, _)| (r, c) == (2, 1)));
        let (_, _, diagonal) = f.blocks().find(|&(r, c, _)| (r, c) == (2, 2)).unwrap();
        let expected = if method == IncompleteMethod::Ict {
            3.75_f64.sqrt()
        } else {
            3.5
        };
        assert!((diagonal[0] - expected).abs() < 1e-14);
        assert_eq!(f.diagnostics().dropped_blocks(), 1);
    }
}

fn fill_reach<S: NumericScalar>(complex: bool) {
    // Eliminating column zero reaches every earlier row. Original columns
    // arrive before fill, so a FIFO traversal would use the wrong pivot order.
    let mut builder = SparseBuilder::new(8, 8);
    for i in 0..8 {
        builder.push(i, i, scalar::<S>(16., 0.)).unwrap();
        if i > 0 {
            let value = scalar::<S>(
                if i == 3 { 0. } else { 1. },
                if complex && i != 3 { 0.25 } else { 0. },
            );
            builder.push(0, i, value).unwrap();
            builder.push(i, 0, value.conj()).unwrap();
        }
    }
    let a = builder
        .finish_csr(DuplicatePolicy::Reject, ExplicitZeroPolicy::Preserve)
        .unwrap();
    let x: Vec<S> = (0..8).map(|i| scalar(i as f64 / 8. + 1., 0.)).collect();
    let mut rhs = vec![S::ZERO; 8];
    for (r, c, v) in a.view().iter() {
        rhs[r] = rhs[r] + v * x[c];
    }
    for method in [
        IncompleteMethod::Ilu { level: 1 },
        IncompleteMethod::Ilut,
        IncompleteMethod::Ict,
    ] {
        for b in [1, 2, 4, 8] {
            let p =
                IncompleteSymbolic::analyze(a.structural_pattern(), method, b, limits()).unwrap();
            let f = IncompleteFactor::factor(p.clone(), a.view(), options()).unwrap();
            let actual = f.solve(&rhs).unwrap();
            let tolerance = S::Real::EPSILON * S::Real::try_from_f64_lossy(128.).unwrap();
            for (&value, &expected) in actual.iter().zip(&x) {
                assert!(
                    (value - expected).abs() < tolerance,
                    "{method:?}, block {b}"
                );
            }
            if b == 1 {
                assert_eq!(
                    f.diagnostics().factor_blocks(),
                    if method == IncompleteMethod::Ict {
                        36
                    } else {
                        64
                    }
                );
            }
        }
    }
}

#[test]
fn dynamically_discovered_fill_and_structural_zeros_preserve_order() {
    fill_reach::<f32>(false);
    fill_reach::<f64>(false);
    fill_reach::<Complex<f32>>(true);
    fill_reach::<Complex<f64>>(true);
}

#[test]
fn every_method_accounts_for_workspace_bytes_and_method_scratch() {
    let a = matrix::<f64>(false);
    for method in [
        IncompleteMethod::Ilu { level: 0 },
        IncompleteMethod::Ilut,
        IncompleteMethod::Ic0,
        IncompleteMethod::Ict,
    ] {
        for b in [1, 2, 4] {
            let p =
                IncompleteSymbolic::analyze(a.structural_pattern(), method, b, limits()).unwrap();
            let bytes = p.workspace_requirements::<f64>(()).unwrap().total_bytes();
            for (budget, success) in [(bytes, true), (bytes - 1, false)] {
                let p = IncompleteSymbolic::analyze(
                    a.structural_pattern(),
                    method,
                    b,
                    limits().with_max_workspace_bytes(budget),
                )
                .unwrap();
                assert_eq!(
                    IncompleteFactor::<f64>::factor(p.clone(), a.view(), options()).is_ok(),
                    success
                );
                assert_eq!(p.workspace::<f64>(()).is_ok(), success);
            }
        }
    }
    let ilu = IncompleteSymbolic::analyze(
        a.structural_pattern(),
        IncompleteMethod::Ilu { level: 0 },
        1,
        limits(),
    )
    .unwrap();
    let ict =
        IncompleteSymbolic::analyze(a.structural_pattern(), IncompleteMethod::Ict, 1, limits())
            .unwrap();
    let mut workspace = ilu.workspace(()).unwrap();
    let mut f = IncompleteFactor::<f64>::factor(ict, a.view(), options()).unwrap();
    assert!(
        f.refactor(
            a.view(),
            scioxide_sparse::SparseSolveOptions::new(),
            &mut workspace,
            ExecutionContext::sequential()
        )
        .is_err()
    );
}

#[test]
fn capacity_failure_preserves_epoch_and_scratch_can_be_reused() {
    // The same pattern with weak and strong couplings: dropping keeps only the
    // diagonals of `a`, while `b` retains off-diagonals beyond four blocks.
    let a = matrix::<f64>(false);
    let b = CsrMatrix::from_raw(
        4,
        4,
        a.row_ptr().to_vec(),
        a.col_indices().to_vec(),
        a.iter()
            .map(|(r, c, v)| if r == c { 10. } else { 6. * v.signum() })
            .collect(),
    )
    .unwrap();
    for method in [IncompleteMethod::Ilut, IncompleteMethod::Ict] {
        let p = IncompleteSymbolic::analyze(
            a.structural_pattern(),
            method,
            1,
            limits().with_max_factor_blocks(4),
        )
        .unwrap();
        let mut f = IncompleteFactor::factor(
            p.try_clone().unwrap(),
            a.view(),
            options().with_drop_tolerance(0.5),
        )
        .unwrap();
        let mut w = p.workspace(()).unwrap();
        let previous = f.solve(&[1.; 4]).unwrap();
        let diagnostics = f.diagnostics();
        assert_eq!(diagnostics.factor_blocks(), 4);
        for _ in 0..2 {
            let result = f.refactor(
                b.view(),
                scioxide_sparse::SparseSolveOptions::new(),
                &mut w,
                ExecutionContext::sequential(),
            );
            assert_eq!(
                result.unwrap_err().kind(),
                scioxide_core::SciOxideErrorKind::ResourceLimit
            );
            assert_eq!(
                w.last_attempt().unwrap().outcome(),
                scioxide_core::AttemptOutcome::Failed
            );
            assert_eq!(f.diagnostics(), diagnostics);
            assert_eq!(f.solve(&[1.; 4]).unwrap(), previous);
            f.refactor(
                a.view(),
                scioxide_sparse::SparseSolveOptions::new(),
                &mut w,
                ExecutionContext::sequential(),
            )
            .unwrap();
            assert_eq!(f.diagnostics(), diagnostics);
        }
    }
}

#[test]
fn absent_diagonals_are_completed_and_repaired_explicitly() {
    let a = CsrMatrix::<f64>::from_raw(2, 2, vec![0, 0, 0], vec![], vec![]).unwrap();
    for method in [
        IncompleteMethod::Ilu { level: 0 },
        IncompleteMethod::Ilut,
        IncompleteMethod::Ic0,
        IncompleteMethod::Ict,
    ] {
        let p = IncompleteSymbolic::analyze(a.structural_pattern(), method, 1, limits()).unwrap();
        assert!(IncompleteFactor::factor(p.clone(), a.view(), options()).is_err());
        let mut o = options();
        o = o.with_pivot(IncompletePivotPolicy::Repair {
            absolute: 0.5,
            relative: 0.,
        });
        let f = IncompleteFactor::factor(p.clone(), a.view(), o).unwrap();
        assert_eq!(f.diagnostics().repaired_pivots(), 2);
        assert_eq!(f.diagnostics().factor_blocks(), 2);
        let x = f.solve(&[1.; 2]).unwrap();
        assert!(x.iter().all(|v| (v - 2.).abs() < 1e-14));
    }
}

#[test]
fn hundred_thousand_unknown_tridiagonal_setup_and_apply() {
    const N: usize = 100_000;
    let mut ptr = Vec::with_capacity(N + 1);
    let mut col = Vec::with_capacity(3 * N - 2);
    let mut values = Vec::with_capacity(3 * N - 2);
    ptr.push(0);
    for i in 0..N {
        if i > 0 {
            col.push(i - 1);
            values.push(-1.);
        }
        col.push(i);
        values.push(4.);
        if i + 1 < N {
            col.push(i + 1);
            values.push(-1.);
        }
        ptr.push(col.len());
    }
    let a = CsrMatrix::from_raw(N, N, ptr, col, values).unwrap();
    let value = |i: usize| 1. + (i % 7) as f64 / 7.;
    let rhs: Vec<_> = (0..N)
        .map(|i| {
            4. * value(i)
                - if i > 0 { value(i - 1) } else { 0. }
                - if i + 1 < N { value(i + 1) } else { 0. }
        })
        .collect();
    for (method, blocks) in [
        (IncompleteMethod::Ilu { level: 0 }, 3 * N - 2),
        (IncompleteMethod::Ilut, 3 * N - 2),
        (IncompleteMethod::Ic0, 2 * N - 1),
        (IncompleteMethod::Ict, 2 * N - 1),
    ] {
        let p = IncompleteSymbolic::analyze(
            a.structural_pattern(),
            method,
            1,
            IncompleteLimits::new(blocks, 128 * 1024 * 1024),
        )
        .unwrap();
        let f = IncompleteFactor::factor(p.clone(), a.view(), options()).unwrap();
        assert_eq!(f.diagnostics().factor_blocks(), blocks);
        let x = f.solve(&rhs).unwrap();
        for (i, &actual) in x.iter().enumerate() {
            assert!((actual - value(i)).abs() < 2e-12, "{method:?}: row {i}");
        }
    }
}
fn exact<S: NumericScalar>(complex: bool, accuracy: Accuracy) {
    let execution = SparseSolveOptions::new().with_accuracy(accuracy);
    let a = matrix::<S>(complex);
    let expected = [
        scalar::<S>(1., 0.),
        scalar(-2., 0.),
        scalar(0.5, 0.),
        scalar(3., 0.),
    ];
    for method in [
        IncompleteMethod::Ilu { level: 3 },
        IncompleteMethod::Ilut,
        IncompleteMethod::Ict,
    ] {
        for b in [1, 2, 4] {
            let p =
                IncompleteSymbolic::analyze(a.structural_pattern(), method, b, limits()).unwrap();
            let f = IncompleteFactor::factor_scoped(
                p.clone(),
                a.view(),
                options(),
                execution,
                ExecutionContext::sequential(),
            )
            .unwrap();
            for op in [
                SparseOperation::Normal,
                SparseOperation::Transpose,
                SparseOperation::Adjoint,
            ] {
                let mut rhs = vec![S::ZERO; 4];
                for (r, c, v) in a.view().iter() {
                    match op {
                        SparseOperation::Normal => rhs[r] = rhs[r] + v * expected[c],
                        SparseOperation::Transpose => rhs[c] = rhs[c] + v * expected[r],
                        SparseOperation::Adjoint => rhs[c] = rhs[c] + v.conj() * expected[r],
                    }
                }
                let mut x = vec![S::ZERO; 4];
                let mut scratch = vec![S::ZERO; f.workspace_requirements(1).unwrap().scalar_len()];
                f.solve_into(
                    &rhs,
                    op,
                    execution,
                    &mut x,
                    &mut scioxide_sparse::SparseSolveWorkspace::borrowed(&mut scratch),
                    ExecutionContext::sequential(),
                )
                .unwrap();
                let tol = S::Real::EPSILON * S::Real::try_from_f64_lossy(256.).unwrap();
                for (v, want) in x.iter().zip(expected) {
                    assert!(
                        (*v - want).abs() < tol,
                        "{method:?}, block {b}, {op:?}: {x:?}"
                    );
                }
            }
        }
    }
}
#[test]
fn all_domains_operations_and_blocks() {
    for accuracy in [Accuracy::Standard, Accuracy::Exact] {
        exact::<f32>(false, accuracy);
        exact::<f64>(false, accuracy);
        exact::<Complex<f32>>(true, accuracy);
        exact::<Complex<f64>>(true, accuracy);
    }
}
#[test]
fn block_lu_handles_zero_scalar_pivot() {
    let a =
        CsrMatrix::from_raw(2, 2, vec![0, 2, 4], vec![0, 1, 0, 1], vec![0., 2., 1., 3.]).unwrap();
    let p =
        IncompleteSymbolic::analyze(a.structural_pattern(), IncompleteMethod::Ilut, 2, limits())
            .unwrap();
    let f = IncompleteFactor::factor(p, a.view(), options()).unwrap();
    assert_eq!(f.diagonal_pivots(0).unwrap(), &[1, 1]);
    for op in [
        SparseOperation::Normal,
        SparseOperation::Transpose,
        SparseOperation::Adjoint,
    ] {
        let rhs = if op == SparseOperation::Normal {
            [4., 7.]
        } else {
            [2., 8.]
        };
        let x = match op {
            SparseOperation::Normal => f.solve(&rhs),
            SparseOperation::Transpose => f.solve_transpose(&rhs),
            SparseOperation::Adjoint => f.solve_adjoint(&rhs),
        }
        .unwrap();
        assert!((x[0] - 1.).abs() < 1e-12);
        assert!((x[1] - 2.).abs() < 1e-12);
    }
}
#[test]
fn levels_dropping_and_original_pattern() {
    let a = matrix::<f64>(false);
    let mut counts = vec![];
    for method in [
        IncompleteMethod::Ilu { level: 0 },
        IncompleteMethod::Ilu { level: 1 },
        IncompleteMethod::Ic0,
        IncompleteMethod::Ict,
    ] {
        let p = IncompleteSymbolic::analyze(a.structural_pattern(), method, 1, limits()).unwrap();
        let f = IncompleteFactor::factor(p.clone(), a.view(), options()).unwrap();
        counts.push(f.diagnostics().factor_blocks());
    }
    assert_eq!(counts[0], 12);
    assert_eq!(counts[1], 14);
    assert_eq!(counts[2], 8);
    assert_eq!(counts[3], 9);
    for method in [IncompleteMethod::Ilut, IncompleteMethod::Ict] {
        let p = IncompleteSymbolic::analyze(a.structural_pattern(), method, 1, limits()).unwrap();
        let mut o = options();
        o = o.with_max_off_diagonal(0);
        let f = IncompleteFactor::factor(p.clone(), a.view(), o).unwrap();
        assert_eq!(f.diagnostics().factor_blocks(), 4);
        assert!(f.diagnostics().dropped_blocks() > 0);
    }
}
#[test]
fn failures_are_transactional_and_budgets_exact() {
    let a = matrix::<f64>(false);
    let p = IncompleteSymbolic::analyze(a.structural_pattern(), IncompleteMethod::Ict, 2, limits())
        .unwrap();
    let mut f = IncompleteFactor::factor(p.clone(), a.view(), options()).unwrap();
    let mut w = p.workspace(()).unwrap();
    let before = f.solve(&[1.; 4]).unwrap();
    let bad = CsrMatrix::from_raw(
        4,
        4,
        a.row_ptr().to_vec(),
        a.col_indices().to_vec(),
        a.values().iter().map(|v| -v).collect(),
    )
    .unwrap();
    assert!(
        f.refactor(
            bad.view(),
            scioxide_sparse::SparseSolveOptions::new(),
            &mut w,
            ExecutionContext::sequential()
        )
        .is_err()
    );
    assert_eq!(f.solve(&[1.; 4]).unwrap(), before);
    let mut output = [7.; 4];
    assert!(f.precondition_into(&[1.; 4], &mut output, &mut []).is_err());
    assert_eq!(output, [7.; 4]);
    let bytes = p.workspace_requirements::<f64>(()).unwrap().total_bytes();
    for (limit, success) in [(bytes, true), (bytes - 1, false)] {
        let mut l = limits();
        l = l.with_max_workspace_bytes(limit);
        let p = IncompleteSymbolic::analyze(a.structural_pattern(), IncompleteMethod::Ict, 2, l)
            .unwrap();
        assert_eq!(
            IncompleteFactor::<f64>::factor(p.clone(), a.view(), options()).is_ok(),
            success
        );
    }
    let mut l = limits();
    l = l.with_max_factor_blocks(2);
    let p =
        IncompleteSymbolic::analyze(a.structural_pattern(), IncompleteMethod::Ict, 2, l).unwrap();
    assert!(IncompleteFactor::factor(p, a.view(), options()).is_err());
}
#[test]
fn repair_and_hermitian_validation() {
    for method in [IncompleteMethod::Ilut, IncompleteMethod::Ict] {
        let a = CsrMatrix::from_raw(2, 2, vec![0, 1, 2], vec![0, 1], vec![0., 2.]).unwrap();
        let p = IncompleteSymbolic::analyze(a.structural_pattern(), method, 1, limits()).unwrap();
        assert!(IncompleteFactor::factor(p.clone(), a.view(), options()).is_err());
        let mut o = options();
        o = o.with_pivot(IncompletePivotPolicy::Repair {
            absolute: 0.5,
            relative: 0.,
        });
        let f = IncompleteFactor::factor(p.clone(), a.view(), o).unwrap();
        assert_eq!(f.diagnostics().repaired_pivots(), 1);
        let x = f.solve(&[1., 2.]).unwrap();
        assert!((x[0] - 2.).abs() < 1e-12);
    }
    let a = CsrMatrix::from_raw(2, 2, vec![0, 2, 3], vec![0, 1, 1], vec![2., 1., 2.]).unwrap();
    let p = IncompleteSymbolic::analyze(a.structural_pattern(), IncompleteMethod::Ict, 1, limits())
        .unwrap();
    assert!(IncompleteFactor::factor(p, a.view(), options()).is_err());
}

#[test]
fn execution_formats_empty_and_work_budgets() {
    use scioxide_core::{
        BlockVectorView, BlockVectorViewMut, Budget, BudgetKind, CancellationToken, ProblemToken,
        SolveBudgets, SolveIdentity, SolveState,
    };
    let a = matrix::<f64>(false);
    let csc = a.to_csc().unwrap();
    let p =
        IncompleteSymbolic::analyze(a.structural_pattern(), IncompleteMethod::Ilut, 2, limits())
            .unwrap();
    let mut f = IncompleteFactor::factor(p.clone(), a.view(), options()).unwrap();
    let mut w = p.workspace(()).unwrap();
    let before = f.solve(&[1.; 4]).unwrap();
    f.refactor(
        csc.view(),
        scioxide_sparse::SparseSolveOptions::new(),
        &mut w,
        ExecutionContext::sequential(),
    )
    .unwrap();
    assert_eq!(before, f.solve(&[1.; 4]).unwrap());
    let input = [1.; 12];
    let mut out = [0.; 12];
    let mut scratch = [0.; 24];
    for op in [
        SparseOperation::Normal,
        SparseOperation::Transpose,
        SparseOperation::Adjoint,
    ] {
        let solve_options = SparseSolveOptions::new();
        f.solve_block_into(
            BlockVectorView::new(&input, 4, 3).unwrap(),
            op,
            solve_options,
            &mut BlockVectorViewMut::new(&mut out, 4, 3).unwrap(),
            &mut scioxide_sparse::SparseSolveWorkspace::borrowed(&mut scratch),
            ExecutionContext::sequential(),
        )
        .unwrap();
        #[cfg(feature = "std")]
        {
            let sequential = out;
            let workers = core::num::NonZeroUsize::new(2).unwrap();
            let executor = scioxide_core::ThreadExecutor::new(workers);
            f.solve_block_into(
                BlockVectorView::new(&input, 4, 3).unwrap(),
                op,
                SparseSolveOptions::new()
                    .with_execution_policy(scioxide_core::ExecutionPolicy::Parallel(workers)),
                &mut BlockVectorViewMut::new(&mut out, 4, 3).unwrap(),
                &mut scioxide_sparse::SparseSolveWorkspace::borrowed(&mut scratch),
                ExecutionContext::new(&executor),
            )
            .unwrap();
            assert_eq!(out, sequential);
        }
    }
    let token = CancellationToken::new();
    token.cancel();
    let prior = out;
    assert!(
        f.solve_block_into(
            BlockVectorView::new(&input, 4, 3).unwrap(),
            SparseOperation::Normal,
            Default::default(),
            &mut BlockVectorViewMut::new(&mut out, 4, 3).unwrap(),
            &mut scioxide_sparse::SparseSolveWorkspace::borrowed(&mut scratch),
            ExecutionContext::sequential().with_cancellation(&token)
        )
        .is_err()
    );
    assert_eq!(out, prior);
    // Each block row admits one iteration; a child scope keeps that allowance
    // local, so one iteration cannot refactor four rows.
    let limited = scioxide_sparse::SparseSolveOptions::new()
        .with_budgets(SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::ONE));
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let identity = SolveIdentity::new(
        ProblemToken::new(489),
        scioxide_core::algorithm_id!("sparse", "incomplete", 1),
        0,
        0,
    );
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
        .unwrap();
    assert!(matches!(
        f.refactor(a.view(), limited, &mut w, &mut context)
            .map_err(|e| e.kind()),
        Err(scioxide_core::SciOxideErrorKind::ConvergenceFailure)
    ));
    assert!(context.counts().nested().iterations() > 0);
    assert_eq!(before, f.solve(&[1.; 4]).unwrap());
    let a = CsrMatrix::<f64>::from_raw(0, 0, vec![0], vec![], vec![]).unwrap();
    let p = IncompleteSymbolic::analyze(
        a.structural_pattern(),
        IncompleteMethod::Ict,
        1,
        IncompleteLimits::new(0, 100),
    )
    .unwrap();
    let f = IncompleteFactor::factor(p, a.view(), options()).unwrap();
    assert_eq!(f.diagnostics().factor_blocks(), 0);
    f.precondition_into(&[], &mut [], &mut []).unwrap();
}
#[test]
fn relative_dropping_is_row_scale_invariant_and_ties_are_stable() {
    let a = CsrMatrix::from_raw(
        3,
        3,
        vec![0, 3, 4, 5],
        vec![0, 1, 2, 1, 2],
        vec![4., 1., 1., 4., 4.],
    )
    .unwrap();
    let p =
        IncompleteSymbolic::analyze(a.structural_pattern(), IncompleteMethod::Ilut, 1, limits())
            .unwrap();
    let mut o = options();
    o = o.with_max_off_diagonal(1);
    o = o.with_drop_tolerance(0.1);
    let f = IncompleteFactor::factor(p.clone(), a.view(), o).unwrap();
    let coordinates: Vec<_> = f.blocks().map(|(r, c, _)| (r, c)).collect();
    assert!(coordinates.contains(&(0, 1)));
    assert!(!coordinates.contains(&(0, 2)));
    for scale in [1e-200, 1e200] {
        let mut scaled = a.clone();
        let mut values = scaled.workspace(()).unwrap();
        scaled
            .update_values(
                |old, next| {
                    for (to, &from) in next.iter_mut().zip(old) {
                        *to = from * scale;
                    }
                    Ok::<_, scioxide_core::SciOxideError>(())
                },
                &mut values,
            )
            .unwrap();
        let f = IncompleteFactor::factor(p.clone(), scaled.view(), o).unwrap();
        assert_eq!(
            coordinates,
            f.blocks().map(|(r, c, _)| (r, c)).collect::<Vec<_>>()
        );
    }
}

#[test]
fn bsr_and_expanded_csr_agree_and_invalid_epochs_do_not_publish() {
    let a = matrix::<f64>(false);
    let bsr = BsrMatrix::from_sparse(a.view(), 2, 2).unwrap();
    let expanded = bsr.to_csr(ExplicitZeroPolicy::Preserve).unwrap();
    for method in [
        IncompleteMethod::Ilu { level: 0 },
        IncompleteMethod::Ilut,
        IncompleteMethod::Ic0,
        IncompleteMethod::Ict,
    ] {
        let p = IncompleteSymbolic::analyze(expanded.structural_pattern(), method, 2, limits())
            .unwrap();
        let mut f = IncompleteFactor::<f64>::factor(p.clone(), bsr.view(), options()).unwrap();
        let mut out = [7.; 4];
        assert!(f.precondition_into(&[1.; 4], &mut out, &mut []).is_err());
        assert_eq!(out, [7.; 4]);
        let mut w = p.workspace(()).unwrap();
        f.refactor(
            bsr.view(),
            scioxide_sparse::SparseSolveOptions::new(),
            &mut w,
            ExecutionContext::sequential(),
        )
        .unwrap();
        let before = f.solve(&[1.; 4]).unwrap();
        f.refactor(
            expanded.view(),
            scioxide_sparse::SparseSolveOptions::new(),
            &mut w,
            ExecutionContext::sequential(),
        )
        .unwrap();
        assert_eq!(before, f.solve(&[1.; 4]).unwrap());
        assert!(
            f.refactor(
                a.view(),
                scioxide_sparse::SparseSolveOptions::new(),
                &mut w,
                ExecutionContext::sequential()
            )
            .is_err()
        );
        // Options are validated once, when a factor is created.
        assert!(
            IncompleteFactor::factor(
                f.symbolic().try_clone().unwrap(),
                bsr.view(),
                options().with_drop_tolerance(f64::NAN)
            )
            .is_err()
        );
        assert_eq!(before, f.solve(&[1.; 4]).unwrap());
        let mut input = [1.; 4];
        input[2] = f64::INFINITY;
        assert!(f.precondition_into(&input, &mut out, &mut [0.; 8]).is_err());
        assert_eq!(out, [7.; 4]);
    }
    assert!(
        IncompleteSymbolic::analyze(a.structural_pattern(), IncompleteMethod::Ilut, 3, limits())
            .is_err()
    );
    let p = IncompleteSymbolic::analyze(
        a.structural_pattern(),
        IncompleteMethod::Ilut,
        1,
        IncompleteLimits::new(usize::MAX, usize::MAX),
    )
    .unwrap();
    assert!(
        matches ! (& (p . workspace_requirements :: < f64 > (())) , Err (__scioxide_error) if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: SizeOverflow { .. }))
    );
}

#[test]
fn omitted_executor_partition_preserves_all_incomplete_factor_outputs() {
    use core::num::NonZeroUsize;
    use scioxide_core::{
        BlockVectorView, BlockVectorViewMut, ExecutionPlan, ExecutionPolicy, Executor, PartitionFn,
        SciOxideError, SciOxideErrorKind,
    };
    struct OmitLast;
    impl Executor for OmitLast {
        fn worker_budget(&self) -> NonZeroUsize {
            NonZeroUsize::new(2).unwrap()
        }
        fn execute(
            &self,
            context: ExecutionContext<'_>,
            plan: ExecutionPlan,
            run: &PartitionFn<'_>,
        ) -> Result<(), SciOxideError> {
            for partition in plan.partitions().take(1) {
                run(partition, (), context)?;
            }
            Ok(())
        }
    }
    let a = matrix::<f64>(false);
    let rhs = [1.; 12];
    for method in [
        IncompleteMethod::Ilu { level: 0 },
        IncompleteMethod::Ilut,
        IncompleteMethod::Ict,
    ] {
        let plan =
            IncompleteSymbolic::analyze(a.structural_pattern(), method, 1, limits()).unwrap();
        let factor = IncompleteFactor::factor(plan.clone(), a.view(), options()).unwrap();
        let mut scratch = vec![0.; factor.workspace_requirements(3).unwrap().scalar_len() + 1];
        let last = scratch.len() - 1;
        scratch[last] = 29.;
        for operation in [
            SparseOperation::Normal,
            SparseOperation::Transpose,
            SparseOperation::Adjoint,
        ] {
            let mut output = [19.; 12];
            let workers = NonZeroUsize::new(2).unwrap();
            let result = factor.solve_block_into(
                BlockVectorView::new(&rhs, 4, 3).unwrap(),
                operation,
                SparseSolveOptions::new().with_execution_policy(ExecutionPolicy::Parallel(workers)),
                &mut BlockVectorViewMut::new(&mut output, 4, 3).unwrap(),
                &mut scioxide_sparse::SparseSolveWorkspace::borrowed(&mut scratch),
                ExecutionContext::new(&OmitLast),
            );
            assert_eq!(
                result.unwrap_err().kind(),
                SciOxideErrorKind::ExecutionFailure
            );
            assert_eq!(output, [19.; 12]);
            assert_eq!(scratch[last], 29.);
        }
    }
}
