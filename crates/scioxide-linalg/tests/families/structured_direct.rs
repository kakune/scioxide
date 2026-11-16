#![allow(clippy::unwrap_used, clippy::expect_used, clippy::needless_range_loop)]
use scioxide_core::LinearOperator;
use scioxide_core::{
    AttemptOutcome, Budget, BudgetKind, Complex, ExecutionContext, NumericScalar, ProblemToken,
    RealScalar, SciOxideErrorKind, SolveBudgets, SolveIdentity, SolveState,
};
use scioxide_core::{DiagonalKind, Triangle};
use scioxide_linalg::*;

#[test]
fn exact_banded_cholesky_rank_preserves_tiny_positive_pivots() {
    let diagonal = [f64::from_bits(1), f64::MIN_POSITIVE, 1., f64::MAX / 4.];
    let matrix = HermitianBandMatrixView::new(4, 0, Triangle::Lower, &diagonal).unwrap();
    let factor = BandedCholeskyDecomposition::factor(matrix).unwrap();
    let exact = DirectSolveOptions::new(RankPolicy::Exact).with_residual(Estimation::Estimate);
    let result = factor
        .solve_with_options(StructuredOperation::Normal, &diagonal[..], &exact)
        .unwrap();
    let rank = result.report().rank();
    assert_eq!(rank.rank(), 4);
    assert_eq!(rank.threshold(), 0.);
    assert!((rank.scale() / diagonal[3] - 1.).abs() < 4. * f64::EPSILON);
    for &value in result.solution().unwrap().as_slice() {
        assert!((value - 1.).abs() < 4. * f64::EPSILON);
    }
}

#[test]
fn narrow_cholesky_batches_retain_exact_failure_counts() {
    let n = 64;
    let mut coefficients = vec![0.; 3 * n];
    for column in coefficients.chunks_exact_mut(3) {
        column[0] = 1.;
    }
    coefficients[40 * 3] = 0.;
    let matrix = HermitianBandMatrixView::new(n, 2, Triangle::Lower, &coefficients).unwrap();
    for (limit, expected) in [(None, 41), (Some(37), 37)] {
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded();
        let identity = SolveIdentity::new(
            ProblemToken::new(2197),
            scioxide_core::algorithm_id!("linalg", "narrow_cholesky_test", 1),
            n as u128,
            0,
        );
        let mut context = state
            .begin_solve(
                ExecutionContext::sequential_native(),
                identity,
                &mut budgets,
            )
            .unwrap();
        let options = limit.map_or_else(LinalgExecutionOptions::new, |limit| {
            LinalgExecutionOptions::new().with_iteration_budget(Budget::try_from(limit).unwrap())
        });
        let mut factor = BandedCholeskyDecomposition::empty();
        let error = BandedCholeskyDecomposition::factor_into(
            matrix,
            &options,
            &mut factor,
            &mut LinalgWorkspace::new(),
            &mut context,
        )
        .unwrap_err();
        assert_eq!(context.counts().nested().iterations(), expected);
        assert_eq!(factor.order(), 0);
        assert_eq!(
            error.kind(),
            if limit.is_some() {
                SciOxideErrorKind::ConvergenceFailure
            } else {
                SciOxideErrorKind::DomainError
            }
        );
    }
}

fn scalar<S: NumericScalar>(x: f64, y: f64) -> S {
    S::from_components(
        S::Real::try_from_f64_exact(x).unwrap(),
        S::Real::try_from_f64_exact(y).unwrap(),
    )
}
fn band<S: NumericScalar>(a: &Matrix<S>, kl: usize, ku: usize) -> BandedMatrix<S> {
    let n = a.rows();
    let mut data = vec![S::ZERO; n * (kl + ku + 1)];
    for j in 0..n {
        for i in j.saturating_sub(ku)..(j + kl + 1).min(n) {
            data[j * (kl + ku + 1) + ku + i - j] = a.view().get(i, j).unwrap();
        }
    }
    BandedMatrix::new(n, n, kl, ku, if n == 0 { 0 } else { kl + ku + 1 }, data).unwrap()
}
fn rhs<S: NumericScalar>(a: &Matrix<S>, op: StructuredOperation) -> Matrix<S> {
    let n = a.rows();
    let mut b = Matrix::zeros(n, 4).unwrap();
    for i in 0..n {
        for c in 0..4 {
            let mut sum = S::ZERO;
            for j in 0..n {
                let v = match op {
                    StructuredOperation::Normal => a.view().get(i, j).unwrap(),
                    StructuredOperation::Transpose => a.view().get(j, i).unwrap(),
                    StructuredOperation::Adjoint => a.view().get(j, i).unwrap().conj(),
                };
                sum = sum + v * scalar::<S>((j + c + 1) as f64, 0.25);
            }
            b.view_mut().set(i, c, sum).unwrap();
        }
    }
    b
}
fn check<S: NumericScalar>(
    a: &Matrix<S>,
    b: &Matrix<S>,
    x: &Matrix<S>,
    op: StructuredOperation,
    tol: f64,
) {
    // Independent dense multiplication, not the structured operator/factor kernel.
    let mut residual = 0.0_f64;
    let mut scale = 0.0_f64;
    for i in 0..a.rows() {
        for c in 0..b.cols() {
            let mut sum = S::ZERO;
            let mut row_scale = 0.0;
            for j in 0..a.cols() {
                let v = match op {
                    StructuredOperation::Normal => a.view().get(i, j).unwrap(),
                    StructuredOperation::Transpose => a.view().get(j, i).unwrap(),
                    StructuredOperation::Adjoint => a.view().get(j, i).unwrap().conj(),
                };
                let product = v * x.view().get(j, c).unwrap();
                sum = sum + product;
                row_scale += product.abs().try_to_f64_exact().unwrap();
            }
            residual = residual.max(
                (sum - b.view().get(i, c).unwrap())
                    .abs()
                    .try_to_f64_exact()
                    .unwrap(),
            );
            scale = scale.max(
                row_scale
                    + b.view()
                        .get(i, c)
                        .unwrap()
                        .abs()
                        .try_to_f64_exact()
                        .unwrap(),
            );
        }
    }
    assert!(
        residual <= tol * scale.max(1.0),
        "residual {residual}, scale {scale}"
    );
}
const OPERATIONS: [StructuredOperation; 3] = [
    StructuredOperation::Normal,
    StructuredOperation::Transpose,
    StructuredOperation::Adjoint,
];
fn qualify<S: NumericScalar>(tol: f64) {
    for n in [0_usize, 1, 2, 7, 16] {
        for kl in 0..n.max(1) {
            for ku in [0, n.saturating_sub(1) / 2, n.saturating_sub(1)] {
                let mut a = Matrix::zeros(n, n).unwrap();
                for j in 0..n {
                    for i in j.saturating_sub(ku)..(j + kl + 1).min(n) {
                        let v = if i == j {
                            scalar::<S>(32.0, 0.0)
                        } else {
                            scalar::<S>(((i * 13 + j * 7) % 5) as f64 - 2.0, 0.125)
                        };
                        a.view_mut().set(i, j, v).unwrap();
                    }
                }
                // Force a row exchange and extra U fill while keeping a nonsingular example.
                if kl > 0 && n > 1 {
                    a.view_mut().set(0, 0, scalar(0.125, 0.0)).unwrap();
                    a.view_mut().set(1, 0, scalar(8.0, 0.25)).unwrap();
                }
                let compact = band(&a, kl, ku);
                let factor = BandedLuDecomposition::factor(&compact).unwrap();
                assert_eq!(factor.factors().len(), n * (2 * kl + ku + 1));
                for op in OPERATIONS {
                    let b = rhs(&a, op);
                    let x = factor.solve_matrix(op, &b).unwrap();
                    check(&a, &b, &x, op, tol);
                }
            }
        }
    }
    for n in [0_usize, 1, 2, 8] {
        for k in 0..n.max(1) {
            let mut l = Matrix::zeros(n, n).unwrap();
            for j in 0..n {
                for i in j..(j + k + 1).min(n) {
                    l.view_mut()
                        .set(
                            i,
                            j,
                            if i == j {
                                scalar::<S>(4.0, 0.0)
                            } else {
                                scalar::<S>(0.25, 0.125)
                            },
                        )
                        .unwrap();
                }
            }
            let mut a = Matrix::zeros(n, n).unwrap();
            for i in 0..n {
                for j in 0..n {
                    let sum = (0..n).fold(S::ZERO, |v, c| {
                        v + l.view().get(i, c).unwrap() * l.view().get(j, c).unwrap().conj()
                    });
                    a.view_mut().set(i, j, sum).unwrap();
                }
            }
            for triangle in [Triangle::Lower, Triangle::Upper] {
                let mut data = vec![S::ZERO; n * (k + 1)];
                for j in 0..n {
                    for i in 0..n {
                        if i.abs_diff(j) <= k
                            && ((triangle == Triangle::Lower && i >= j)
                                || (triangle == Triangle::Upper && i <= j))
                        {
                            let offset = if triangle == Triangle::Lower {
                                i - j
                            } else {
                                k + i - j
                            };
                            data[j * (k + 1) + offset] = a.view().get(i, j).unwrap();
                        }
                    }
                }
                let compact = HermitianBandMatrix::new(n, k, triangle, data).unwrap();
                let factor = BandedCholeskyDecomposition::factor(&compact).unwrap();
                let lower = factor.lower().unwrap();
                for i in 0..n {
                    for j in 0..n {
                        assert!(
                            (lower.get(i, j).unwrap() - l.view().get(i, j).unwrap())
                                .abs()
                                .try_to_f64_exact()
                                .unwrap()
                                < tol
                        );
                    }
                }
                for op in OPERATIONS {
                    let b = rhs(&a, op);
                    let x = factor.solve_matrix(op, &b).unwrap();
                    check(&a, &b, &x, op, tol);
                }
            }
        }
    }
    for triangle in [Triangle::Lower, Triangle::Upper] {
        for diagonal in [DiagonalKind::Explicit, DiagonalKind::Unit] {
            let n = 7;
            let mut a = Matrix::zeros(n, n).unwrap();
            let mut packed = Vec::new();
            for j in 0..n {
                for i in 0..n {
                    if (triangle == Triangle::Lower && i >= j)
                        || (triangle == Triangle::Upper && i <= j)
                    {
                        let v = if i == j {
                            if diagonal == DiagonalKind::Unit {
                                S::ONE
                            } else {
                                scalar::<S>(4.0, 0.0)
                            }
                        } else {
                            scalar::<S>(0.25, 0.125)
                        };
                        a.view_mut().set(i, j, v).unwrap();
                        packed.push(v);
                    }
                }
            }
            let compact = PackedMatrix::new(n, triangle, PackedKind::Triangular, packed).unwrap();
            let bm = band(&a, n - 1, n - 1);
            let plans = [
                TriangularPlan::from_dense(a.view(), triangle, diagonal).unwrap(),
                TriangularPlan::from_packed(compact.view(), diagonal).unwrap(),
                TriangularPlan::from_band(bm.view(), triangle, diagonal).unwrap(),
            ];
            for plan in plans {
                for op in OPERATIONS {
                    let b = rhs(&a, op);
                    let x = plan.solve_matrix(op, &b).unwrap();
                    check(&a, &b, &x, op, tol);
                }
            }
        }
    }
}
#[test]
fn all_scalar_numerics() {
    qualify::<f32>(2e-5);
    qualify::<f64>(2e-13);
    qualify::<Complex<f32>>(2e-5);
    qualify::<Complex<f64>>(2e-13);
}
#[test]
fn validation_and_atomicity() {
    assert!(BandedMatrixView::<f64>::new(3, 3, 3, 0, 4, &[]).is_err());
    assert!(
        BandedMatrixView::<f64>::new(usize::MAX, usize::MAX, usize::MAX - 1, 0, usize::MAX, &[])
            .is_err()
    );
    assert!(
        PackedMatrixView::<f64>::new(usize::MAX, Triangle::Upper, PackedKind::Triangular, &[])
            .is_err()
    );
    assert!(
        HermitianBandMatrixView::new(1, 0, Triangle::Lower, &[Complex::new(1.0, 1.0)]).is_err()
    );
    let options = LinalgExecutionOptions::new();
    let a = BandedMatrix::new(2, 2, 1, 1, 3, vec![f64::NAN, 4.0, 1.0, 1.0, 4.0, f64::NAN]).unwrap();
    let mut f = BandedLuDecomposition::factor(&a).unwrap();
    let before = f.clone();
    let mut w = LinalgWorkspace::new();
    let singular = BandedMatrix::new(2, 2, 1, 1, 3, vec![0.0; 6]).unwrap();
    assert!(
        BandedLuDecomposition::factor_into(
            &singular,
            &options,
            &mut f,
            &mut w,
            ExecutionContext::sequential()
        )
        .is_err_and(|error| matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::Singular { .. }
        ))
    );
    assert_eq!(f, before);
    let rhs = Matrix::from_vec(2, 1, vec![1.0, f64::NAN]).unwrap();
    let mut out = Matrix::from_vec(2, 1, vec![77.0; 2]).unwrap();
    assert!(
        f.solve_matrix_into(
            StructuredOperation::Normal,
            &rhs,
            &scioxide_linalg::DirectSolveOptions::new(scioxide_linalg::RankPolicy::Exact),
            out.view_mut(),
            &mut w,
            ExecutionContext::sequential()
        )
        .and_then(|report| {
            if report.termination() == scioxide_linalg::DirectSolveTermination::Accepted {
                Ok(())
            } else {
                Err(scioxide_core::SciOxideError::singular(None))
            }
        })
        .is_err()
    );
    assert_eq!(out.as_slice(), &[77.0; 2]);
    assert_eq!(
        w.last_attempt().unwrap().outcome(),
        AttemptOutcome::PreflightRejected
    );
    // Non-finite coefficients are rejected at the boundary, before an epoch.
    let nan = BandedMatrix::new(2, 2, 1, 1, 3, vec![0.0, f64::NAN, 1.0, 1.0, 4.0, 0.0]).unwrap();
    assert_eq!(
        BandedLuDecomposition::factor_into(
            &nan,
            &options,
            &mut f,
            &mut w,
            ExecutionContext::sequential()
        )
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::NonFinite
    );
    assert_eq!(
        w.last_attempt().unwrap().outcome(),
        AttemptOutcome::PreflightRejected
    );
    assert_eq!(f, before);
    // Empty systems succeed without work and record a successful attempt.
    let mut empty = Matrix::zeros(0, 3).unwrap();
    BandedLuDecomposition::<f64>::empty()
        .solve_matrix_into(
            StructuredOperation::Normal,
            &Matrix::zeros(0, 3).unwrap(),
            &scioxide_linalg::DirectSolveOptions::new(scioxide_linalg::RankPolicy::Exact),
            empty.view_mut(),
            &mut w,
            ExecutionContext::sequential(),
        )
        .and_then(|report| {
            if report.termination() == scioxide_linalg::DirectSolveTermination::Accepted {
                Ok(())
            } else {
                Err(scioxide_core::SciOxideError::singular(None))
            }
        })
        .unwrap();
    assert_eq!(
        w.last_attempt().unwrap().outcome(),
        AttemptOutcome::Succeeded
    );
    let indefinite = HermitianBandMatrix::new(1, 0, Triangle::Lower, vec![-1.0]).unwrap();
    assert!(matches!(
        BandedCholeskyDecomposition::factor(&indefinite)
            .unwrap_err()
            .view(),
        scioxide_core::SciOxideErrorView::DomainError { .. }
    ));
}

fn tridiagonal_band(n: usize) -> BandedMatrix<f64> {
    BandedMatrix::new(
        n,
        n,
        1,
        1,
        if n == 0 { 0 } else { 3 },
        (0..n * 3)
            .map(|i| if i % 3 == 1 { 4.0 } else { 1.0 })
            .collect(),
    )
    .unwrap()
}

fn identity(order: usize) -> SolveIdentity {
    SolveIdentity::new(
        ProblemToken::new(472),
        scioxide_core::algorithm_id!("linalg", "structured_test", 1),
        order as u128,
        0,
    )
}

#[test]
fn budget_and_cancellation_preserve_published_state() {
    use scioxide_core::CancellationToken;
    let a = BandedMatrix::new(3, 3, 0, 0, 1, vec![2.0; 3]).unwrap();
    let options = LinalgExecutionOptions::new();
    let mut f = BandedLuDecomposition::factor(&a).unwrap();
    let before = f.clone();
    let mut w = LinalgWorkspace::new();
    // A local iteration allowance bounds the pivot columns of the child epoch.
    let limited = options.with_iteration_budget(Budget::try_from(1).unwrap());
    let error = BandedLuDecomposition::factor_into(
        &a,
        &limited,
        &mut f,
        &mut w,
        ExecutionContext::sequential(),
    )
    .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::ConvergenceFailure);
    assert_eq!(f, before);
    let attempt = w.last_attempt().unwrap();
    assert_eq!(attempt.outcome(), AttemptOutcome::Stopped);
    assert_eq!(attempt.counts().own().iterations(), 1);
    // A parent's scope-local allowance is not consumed by the child epoch.
    let mut budgets =
        SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::try_from(1).unwrap());
    let mut state = SolveState::new();
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), identity(3), &mut budgets)
        .unwrap();
    BandedLuDecomposition::factor_into(&a, &options, &mut f, &mut w, &mut context).unwrap();
    assert_eq!(context.counts().own().iterations(), 0);
    assert_eq!(context.counts().nested().iterations(), 3);
    let rhs = Matrix::from_vec(3, 4, vec![1.0; 12]).unwrap();
    let mut out = Matrix::from_vec(3, 4, vec![99.0; 12]).unwrap();
    f.solve_matrix_into(
        StructuredOperation::Normal,
        &rhs,
        &scioxide_linalg::DirectSolveOptions::new(scioxide_linalg::RankPolicy::Exact),
        out.view_mut(),
        &mut w,
        &mut context,
    )
    .and_then(|report| {
        if report.termination() == scioxide_linalg::DirectSolveTermination::Accepted {
            Ok(())
        } else {
            Err(scioxide_core::SciOxideError::singular(None))
        }
    })
    .unwrap();
    assert_eq!(context.counts().nested().iterations(), 7);
    let attempt = w.last_attempt().unwrap();
    assert_eq!(attempt.outcome(), AttemptOutcome::Succeeded);
    assert_eq!(attempt.counts().own().iterations(), 4);
    assert!(out.as_slice().iter().all(|&x| (x - 0.5).abs() < 1e-15));
    let mut out = Matrix::from_vec(3, 4, vec![99.0; 12]).unwrap();
    let token = CancellationToken::new();
    token.cancel();
    let execution = ExecutionContext::sequential().with_cancellation(&token);
    assert!(BandedLuDecomposition::factor_into(&a, &options, &mut f, &mut w, execution).is_err());
    assert_eq!(f, before);
    assert!(
        f.solve_matrix_into(
            StructuredOperation::Normal,
            &rhs,
            &scioxide_linalg::DirectSolveOptions::new(scioxide_linalg::RankPolicy::Exact),
            out.view_mut(),
            &mut w,
            execution
        )
        .and_then(|report| {
            if report.termination() == scioxide_linalg::DirectSolveTermination::Accepted {
                Ok(())
            } else {
                Err(scioxide_core::SciOxideError::singular(None))
            }
        })
        .is_err()
    );
    assert_eq!(out.as_slice(), &[99.0; 12]);
    assert_eq!(
        w.last_attempt().unwrap().outcome(),
        AttemptOutcome::PreflightRejected
    );
}

#[test]
fn requirements_limits_overflow_and_borrowed_scratch() {
    let options = LinalgExecutionOptions::new();
    let limited = options.with_byte_limit(1);
    let factor = StructuredWorkspaceRequest::factor(4, 1, 1, &limited);
    for result in [
        BandedLuDecomposition::<f64>::workspace_requirements(&factor),
        BandedCholeskyDecomposition::<f64>::workspace_requirements(&factor),
        BandedLuDecomposition::<f64>::workspace_requirements(&StructuredWorkspaceRequest::solve(
            4, &limited,
        )),
        TridiagonalLuDecomposition::<f64>::workspace_requirements(
            &StructuredWorkspaceRequest::factor(4, 1, 1, &limited),
        ),
        TridiagonalLuDecomposition::<f64>::workspace_requirements(
            &StructuredWorkspaceRequest::solve_matrix(4, 2, &limited),
        ),
    ] {
        assert_eq!(result.unwrap_err().kind(), SciOxideErrorKind::ResourceLimit);
    }
    for result in
        [
            BandedLuDecomposition::<f64>::workspace_requirements(
                &StructuredWorkspaceRequest::factor(usize::MAX, 1, 1, &options),
            ),
            BandedCholeskyDecomposition::<f64>::workspace_requirements(
                &StructuredWorkspaceRequest::factor(usize::MAX, 2, 2, &options),
            ),
            BandedLuDecomposition::<f64>::workspace_requirements(
                &StructuredWorkspaceRequest::solve_matrix(usize::MAX, 2, &options),
            ),
            TridiagonalLuDecomposition::<f64>::workspace_requirements(
                &StructuredWorkspaceRequest::factor(usize::MAX, 1, 1, &options),
            ),
        ]
    {
        assert_eq!(result.unwrap_err().kind(), SciOxideErrorKind::SizeOverflow);
    }
    let a = tridiagonal_band(3);
    let mut f = BandedLuDecomposition::factor(&a).unwrap();
    let before = f.clone();
    // Byte limits are enforced before any growth of an owned workspace.
    let mut fresh = LinalgWorkspace::new();
    assert_eq!(
        BandedLuDecomposition::factor_into(
            &a,
            &limited,
            &mut f,
            &mut fresh,
            ExecutionContext::sequential()
        )
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::ResourceLimit
    );
    assert_eq!(fresh.retained_bytes().unwrap(), 0);
    assert_eq!(f, before);
    // Borrowed scratch never grows.
    let mut short = [0.0; 2];
    assert_eq!(
        BandedLuDecomposition::factor_into(
            &a,
            &options,
            &mut f,
            &mut LinalgWorkspace::from_scalar_slice(&mut short),
            ExecutionContext::sequential()
        )
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(f, before);
    let mut output = [5.0; 3];
    assert_eq!(
        f.solve_into(
            StructuredOperation::Normal,
            &[1.0; 3][..],
            &scioxide_linalg::DirectSolveOptions::new(scioxide_linalg::RankPolicy::Exact),
            VectorViewMut::from_slice(&mut output),
            &mut LinalgWorkspace::from_scalar_slice(&mut short),
            ExecutionContext::sequential()
        )
        .and_then(|report| {
            if report.termination() == scioxide_linalg::DirectSolveTermination::Accepted {
                Ok(())
            } else {
                Err(scioxide_core::SciOxideError::singular(None))
            }
        })
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(output, [5.0; 3]);
    let mut enough = [0.0; 3];
    f.solve_into(
        StructuredOperation::Transpose,
        &[5.0, 6.0, 5.0][..],
        &scioxide_linalg::DirectSolveOptions::new(scioxide_linalg::RankPolicy::Exact),
        VectorViewMut::from_slice(&mut output),
        &mut LinalgWorkspace::from_scalar_slice(&mut enough),
        ExecutionContext::sequential(),
    )
    .and_then(|report| {
        if report.termination() == scioxide_linalg::DirectSolveTermination::Accepted {
            Ok(())
        } else {
            Err(scioxide_core::SciOxideError::singular(None))
        }
    })
    .unwrap();
    assert!(output.iter().all(|&x| (x - 1.0).abs() < 1e-14));
    assert_eq!(
        f.solve_into(
            StructuredOperation::Normal,
            &[1.0; 2][..],
            &scioxide_linalg::DirectSolveOptions::new(scioxide_linalg::RankPolicy::Exact),
            VectorViewMut::from_slice(&mut output),
            &mut LinalgWorkspace::new(),
            ExecutionContext::sequential()
        )
        .and_then(|report| {
            if report.termination() == scioxide_linalg::DirectSolveTermination::Accepted {
                Ok(())
            } else {
                Err(scioxide_core::SciOxideError::singular(None))
            }
        })
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::DimensionMismatch
    );
    let tri = TridiagonalMatrix::new(vec![1.0; 2], vec![4.0; 3], vec![1.0; 2]).unwrap();
    let mut t = TridiagonalLuDecomposition::factor(&tri).unwrap();
    let before = t.clone();
    assert_eq!(
        TridiagonalLuDecomposition::factor_into(
            &tri,
            &options,
            &mut t,
            &mut LinalgWorkspace::from_scalar_slice(&mut short),
            ExecutionContext::sequential()
        )
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(t, before);
    let h = HermitianBandMatrix::new(3, 1, Triangle::Lower, vec![4.0, 1.0, 4.0, 1.0, 4.0, 0.0])
        .unwrap();
    let mut c = BandedCholeskyDecomposition::factor(&h).unwrap();
    let before = c.clone();
    assert_eq!(
        BandedCholeskyDecomposition::factor_into(
            &h,
            &limited,
            &mut c,
            &mut LinalgWorkspace::new(),
            ExecutionContext::sequential()
        )
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::ResourceLimit
    );
    assert_eq!(c, before);
    let plan =
        TriangularPlan::from_band(a.view(), Triangle::Upper, DiagonalKind::Explicit).unwrap();
    let requirements =
        |request: StructuredWorkspaceRequest| plan.workspace_requirements(&request).unwrap_err();
    assert_eq!(
        requirements(StructuredWorkspaceRequest::solve_matrix(
            3,
            usize::MAX,
            &options
        ))
        .kind(),
        SciOxideErrorKind::SizeOverflow
    );
    assert_eq!(
        requirements(StructuredWorkspaceRequest::factor(3, 0, 2, &options)).kind(),
        SciOxideErrorKind::InvalidInput
    );
    assert_eq!(
        requirements(StructuredWorkspaceRequest::solve(4, &options)).kind(),
        SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(
        plan.solve_into(
            StructuredOperation::Normal,
            &[1.0; 3][..],
            &DirectSolveOptions::new(RankPolicy::Exact),
            VectorViewMut::from_slice(&mut output),
            &mut LinalgWorkspace::from_scalar_slice(&mut short),
            ExecutionContext::sequential()
        )
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::DimensionMismatch
    );
    let time_limited = options.with_time_limit(core::time::Duration::from_secs(1));
    assert_eq!(
        BandedLuDecomposition::factor_with_options(&a, &time_limited)
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::CapabilityUnavailable
    );
    assert_eq!(
        BandedCholeskyDecomposition::factor_with_options(&h, &time_limited)
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::CapabilityUnavailable
    );
    assert_eq!(
        TridiagonalLuDecomposition::factor_with_options(&tri, &time_limited)
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::CapabilityUnavailable
    );
}

#[cfg(feature = "std")]
#[test]
fn executor_matches_sequential_for_all_families() {
    let pool = scioxide_core::ThreadExecutor::new(core::num::NonZeroUsize::new(3).unwrap());
    let execution = ExecutionContext::new(&pool).with_processor(&scioxide_core::StdProcessor);
    let a = BandedMatrix::new(
        3,
        3,
        1,
        1,
        3,
        vec![0.0, 4.0, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0, 0.0],
    )
    .unwrap();
    let f = BandedLuDecomposition::factor(&a).unwrap();
    let rhs = Matrix::from_vec(3, 4, (0..12).map(|v| v as f64).collect()).unwrap();
    let mut out = Matrix::zeros(3, 4).unwrap();
    let mut w = LinalgWorkspace::new();
    let expected = f.solve_matrix(StructuredOperation::Adjoint, &rhs).unwrap();
    f.solve_matrix_into(
        StructuredOperation::Adjoint,
        &rhs,
        &scioxide_linalg::DirectSolveOptions::new(scioxide_linalg::RankPolicy::Exact),
        out.view_mut(),
        &mut w,
        execution,
    )
    .and_then(|report| {
        if report.termination() == scioxide_linalg::DirectSolveTermination::Accepted {
            Ok(())
        } else {
            Err(scioxide_core::SciOxideError::singular(None))
        }
    })
    .unwrap();
    assert_eq!(out, expected);
    let h = HermitianBandMatrix::new(3, 1, Triangle::Lower, vec![4.0, 1.0, 4.0, 1.0, 4.0, 0.0])
        .unwrap();
    let c = BandedCholeskyDecomposition::factor(&h).unwrap();
    let expected = c.solve_matrix(StructuredOperation::Normal, &rhs).unwrap();
    c.solve_matrix_into(
        StructuredOperation::Normal,
        &rhs,
        &scioxide_linalg::DirectSolveOptions::new(scioxide_linalg::RankPolicy::Exact),
        out.view_mut(),
        &mut w,
        execution,
    )
    .and_then(|report| {
        if report.termination() == scioxide_linalg::DirectSolveTermination::Accepted {
            Ok(())
        } else {
            Err(scioxide_core::SciOxideError::singular(None))
        }
    })
    .unwrap();
    assert_eq!(out, expected);
    let plan =
        TriangularPlan::from_band(a.view(), Triangle::Lower, DiagonalKind::Explicit).unwrap();
    let expected = plan
        .solve_matrix(StructuredOperation::Normal, &rhs)
        .unwrap();
    plan.solve_matrix_into(
        StructuredOperation::Normal,
        &rhs,
        &scioxide_linalg::DirectSolveOptions::new(scioxide_linalg::RankPolicy::Exact),
        out.view_mut(),
        &mut w,
        execution,
    )
    .and_then(|report| {
        if report.termination() == scioxide_linalg::DirectSolveTermination::Accepted {
            Ok(())
        } else {
            Err(scioxide_core::SciOxideError::singular(None))
        }
    })
    .unwrap();
    assert_eq!(out, expected);
    let t = TridiagonalLuDecomposition::factor(
        TridiagonalMatrixView::new(&[1.0; 2], &[4.0; 3], &[1.0; 2]).unwrap(),
    )
    .unwrap();
    let expected = t.solve_matrix(StructuredOperation::Normal, &rhs).unwrap();
    t.solve_matrix_into(
        StructuredOperation::Normal,
        &rhs,
        &scioxide_linalg::DirectSolveOptions::new(scioxide_linalg::RankPolicy::Exact),
        out.view_mut(),
        &mut w,
        execution,
    )
    .and_then(|report| {
        if report.termination() == scioxide_linalg::DirectSolveTermination::Accepted {
            Ok(())
        } else {
            Err(scioxide_core::SciOxideError::singular(None))
        }
    })
    .unwrap();
    assert_eq!(out, expected);
}
#[test]
fn empty_shapes_and_extreme_scales() {
    let empty = BandedLuDecomposition::<f64>::empty();
    let rhs = Matrix::zeros(0, usize::MAX).unwrap();
    let out = empty
        .solve_matrix(StructuredOperation::Normal, &rhs)
        .unwrap();
    assert_eq!(out.cols(), usize::MAX);
    for scale in [f64::from_bits(1), 1e-300, 1e300, f64::MAX] {
        let a = BandedMatrix::new(1, 1, 0, 0, 1, vec![scale]).unwrap();
        let factor = BandedLuDecomposition::factor(&a).unwrap();
        let x = factor
            .solve(StructuredOperation::Normal, &[scale][..])
            .unwrap();
        assert_eq!(x.as_slice(), &[1.0]);
        let h = HermitianBandMatrix::new(1, 0, Triangle::Lower, vec![scale]).unwrap();
        let factor = BandedCholeskyDecomposition::factor(&h).unwrap();
        let x = factor
            .solve(StructuredOperation::Normal, &[scale][..])
            .unwrap();
        assert!((x.as_slice()[0] - 1.0).abs() < 1e-14);
    }
    let t = TridiagonalLuDecomposition::factor(
        TridiagonalMatrixView::new(&[0.0], &[1e-30, 1.0], &[1.0]).unwrap(),
    )
    .unwrap();
    let x = t
        .solve(StructuredOperation::Normal, &[1e-30, 0.0][..])
        .unwrap();
    assert_eq!(x.as_slice(), &[1.0, 0.0]);
}

#[test]
fn lu_reconstruction_with_repeated_distant_pivots() {
    // A permutation-dominated band forces exchanges beyond adjacent rows and
    // fills the upper band. Reverse elementary transformations independently.
    let n = 8;
    let kl = 3;
    let ku = 3;
    let mut a = Matrix::zeros(n, n).unwrap();
    for j in 0..n {
        for i in j.saturating_sub(ku)..(j + kl + 1).min(n) {
            let re = if i == (j + 3) % n {
                8.0
            } else if i == j {
                0.125
            } else {
                0.25
            };
            a.view_mut()
                .set(i, j, Complex::new(re, if i == j { 0.0 } else { 0.0625 }))
                .unwrap();
        }
    }
    let f = BandedLuDecomposition::factor(&band(&a, kl, ku)).unwrap();
    assert!(f.pivots().iter().enumerate().any(|(j, &p)| p > j + 1));
    let width = 2 * kl + ku + 1;
    let diag = kl + ku;
    let mut reconstructed = Matrix::zeros(n, n).unwrap();
    for i in 0..n {
        for j in i..(i + diag + 1).min(n) {
            reconstructed
                .view_mut()
                .set(i, j, f.factors()[j * width + diag + i - j])
                .unwrap();
        }
    }
    for j in (0..n).rev() {
        for i in j + 1..(j + kl + 1).min(n) {
            let multiplier = f.factors()[j * width + diag + i - j];
            for c in 0..n {
                let v = reconstructed.view().get(i, c).unwrap()
                    + multiplier * reconstructed.view().get(j, c).unwrap();
                reconstructed.view_mut().set(i, c, v).unwrap();
            }
        }
        let pivot = f.pivots()[j];
        for c in 0..n {
            let x = reconstructed.view().get(j, c).unwrap();
            let y = reconstructed.view().get(pivot, c).unwrap();
            reconstructed.view_mut().set(j, c, y).unwrap();
            reconstructed.view_mut().set(pivot, c, x).unwrap();
        }
    }
    for (x, y) in reconstructed.as_slice().iter().zip(a.as_slice()) {
        assert!((*x - *y).norm() < 1e-12);
    }
    for op in OPERATIONS {
        let b = rhs(&a, op);
        let x = f.solve_matrix(op, &b).unwrap();
        check(&a, &b, &x, op, 2e-13);
    }
}
#[test]
fn packed_hermitian_and_ignored_unit_diagonal() {
    let a = PackedMatrix::new(
        2,
        Triangle::Lower,
        PackedKind::Hermitian,
        vec![
            Complex::new(4.0, 0.0),
            Complex::new(1.0, 2.0),
            Complex::new(3.0, 0.0),
        ],
    )
    .unwrap();
    assert_eq!(a.view().get(0, 1).unwrap(), Complex::new(1.0, -2.0));
    let mut y = [Complex::new(0.0, 0.0); 2];
    let mut w = y;
    let x = [Complex::new(1.0, 0.0); 2];
    a.view().apply_into(&x, &mut y, &mut w).unwrap();
    assert_eq!(y, [Complex::new(5.0, -2.0), Complex::new(4.0, 2.0)]);
    let packed = PackedMatrix::new(
        2,
        Triangle::Lower,
        PackedKind::Triangular,
        vec![f64::NAN, 2.0, f64::NAN],
    )
    .unwrap();
    let plan = TriangularPlan::from_packed(packed.view(), DiagonalKind::Unit).unwrap();
    assert_eq!(
        plan.solve(StructuredOperation::Normal, &[1.0, 3.0][..])
            .unwrap()
            .as_slice(),
        &[1.0, 1.0]
    );
    assert!(matches!(
        (packed.view().get(2, 0)).as_ref().err().unwrap().view(),
        scioxide_core::SciOxideErrorView::OutOfBounds { .. }
    ));
}
#[test]
fn insufficient_workspace_and_overflow_do_not_publish() {
    let a = BandedMatrix::new(1, 1, 0, 0, 1, vec![f64::from_bits(1)]).unwrap();
    let f = BandedLuDecomposition::factor(&a).unwrap();
    let mut output = Vector::from_vec(vec![17.0]);
    assert!(
        f.solve_into(
            StructuredOperation::Normal,
            &[f64::MAX][..],
            &scioxide_linalg::DirectSolveOptions::new(scioxide_linalg::RankPolicy::Exact),
            output.view_mut(),
            &mut LinalgWorkspace::from_scalar_slice(&mut []),
            ExecutionContext::sequential()
        )
        .and_then(|report| {
            if report.termination() == scioxide_linalg::DirectSolveTermination::Accepted {
                Ok(())
            } else {
                Err(scioxide_core::SciOxideError::singular(None))
            }
        })
        .is_err()
    );
    assert_eq!(output.as_slice(), &[17.0]);
    assert!(matches!(
        f.solve_into(
            StructuredOperation::Normal,
            &[f64::MAX][..],
            &scioxide_linalg::DirectSolveOptions::new(scioxide_linalg::RankPolicy::Exact),
            output.view_mut(),
            &mut LinalgWorkspace::new(),
            ExecutionContext::sequential()
        )
        .and_then(|report| {
            if report.termination() == scioxide_linalg::DirectSolveTermination::Accepted {
                Ok(())
            } else {
                Err(scioxide_core::SciOxideError::singular(None))
            }
        })
        .unwrap_err()
        .view(),
        scioxide_core::SciOxideErrorView::NumericalInstability { .. }
    ));
    assert_eq!(output.as_slice(), &[17.0]);
}

#[test]
fn tridiagonal_distinguishes_factor_invariants_from_solution_rounding() {
    // The multiplier 1e-330 underflows to zero. The
    // factors stay normwise backward stable for this badly scaled matrix.
    let a = TridiagonalMatrixView::new(&[1e-30], &[1e300, 2e-30], &[1e300]).unwrap();
    let factor = TridiagonalLuDecomposition::factor(a).unwrap();
    let options = DirectSolveOptions::new(RankPolicy::Exact).with_residual(Estimation::Estimate);
    let result = factor
        .solve_with_options(StructuredOperation::Normal, &[1e300, 1e-30][..], &options)
        .unwrap();
    assert!(result.report().residual().unwrap().scaled_norm() <= f64::EPSILON);
    let factor =
        TridiagonalLuDecomposition::factor(TridiagonalMatrixView::new(&[], &[1e308], &[]).unwrap())
            .unwrap();
    let rhs = [1e-308];
    for operation in [
        StructuredOperation::Normal,
        StructuredOperation::Transpose,
        StructuredOperation::Adjoint,
    ] {
        let x = factor.solve(operation, &rhs[..]).unwrap();
        assert_eq!(x.as_slice(), &[0.0]);
    }
}

#[test]
fn rectangular_bands_are_rejected_before_factor_or_workspace_changes() {
    let options = LinalgExecutionOptions::new();
    let square = BandedMatrix::new(1, 1, 0, 0, 1, vec![2.0_f64]).unwrap();
    let mut factor = BandedLuDecomposition::factor(&square).unwrap();
    let before = factor.clone();
    let mut workspace = LinalgWorkspace::new();
    let rectangular = BandedMatrix::new(2, 3, 0, 0, 1, vec![1.0, 1.0, 0.0]).unwrap();
    assert!(BandedLuDecomposition::factor(&rectangular).is_err());
    let error = BandedLuDecomposition::factor_into(
        &rectangular,
        &options,
        &mut factor,
        &mut workspace,
        ExecutionContext::sequential(),
    )
    .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::DimensionMismatch);
    assert_eq!(factor, before);
    assert_eq!(workspace.retained_bytes().unwrap(), 0);
    assert!(
        TriangularPlan::from_band(rectangular.view(), Triangle::Lower, DiagonalKind::Unit).is_err()
    );
}

fn wide_bands<S: NumericScalar>() {
    #[cfg(feature = "std")]
    static PROCESSOR: scioxide_core::StdProcessor = scioxide_core::StdProcessor;
    #[cfg(feature = "std")]
    let contexts = [
        ExecutionContext::sequential(),
        ExecutionContext::sequential().with_processor(&PROCESSOR),
    ];
    #[cfg(not(feature = "std"))]
    let contexts = [ExecutionContext::sequential()];
    for execution in contexts {
        let options = LinalgExecutionOptions::new();
        let tolerance = 256.0 * S::Real::EPSILON.try_to_f64_exact().unwrap();
        for (n, kl, ku) in [(35, 16, 3), (65, 31, 31), (67, 32, 32), (103, 50, 50)] {
            for pivoting in [false, true] {
                let mut a = Matrix::zeros(n, n).unwrap();
                for j in 0..n {
                    for i in j.saturating_sub(ku)..(j + kl + 1).min(n) {
                        let re = ((17 * i + 13 * j + 3) % 31) as f64 / 32.0 - 0.5;
                        let im = ((7 * i + 11 * j + 1) % 23) as f64 / 32.0 - 0.5;
                        let value = if i == j {
                            scalar::<S>(if pivoting { 0.25 } else { 64.0 }, 0.0)
                        } else {
                            scalar::<S>(re, im)
                        };
                        a.view_mut().set(i, j, value).unwrap();
                    }
                }
                let mut factor = BandedLuDecomposition::empty();
                BandedLuDecomposition::factor_into(
                    &band(&a, kl, ku),
                    &options,
                    &mut factor,
                    &mut LinalgWorkspace::new(),
                    execution,
                )
                .unwrap();
                if pivoting {
                    assert!(factor.pivots().iter().enumerate().any(|(i, &p)| i != p));
                }
                for op in [
                    StructuredOperation::Normal,
                    StructuredOperation::Transpose,
                    StructuredOperation::Adjoint,
                ] {
                    let b = rhs(&a, op);
                    let x = factor.solve_matrix(op, &b).unwrap();
                    check(&a, &b, &x, op, tolerance);
                }
            }
            let k = kl;
            let mut a = Matrix::zeros(n, n).unwrap();
            let mut lower = vec![S::ZERO; n * (k + 1)];
            for j in 0..n {
                for i in j..(j + k + 1).min(n) {
                    let value = if i == j {
                        scalar::<S>(64.0, 0.0)
                    } else {
                        scalar::<S>(((i + j * 7) % 13) as f64 / 16.0 - 0.5, 0.125)
                    };
                    a.view_mut().set(i, j, value).unwrap();
                    a.view_mut().set(j, i, value.conj()).unwrap();
                    lower[j * (k + 1) + i - j] = value;
                }
            }
            let h = HermitianBandMatrix::new(n, k, Triangle::Lower, lower).unwrap();
            let mut factor = BandedCholeskyDecomposition::empty();
            BandedCholeskyDecomposition::factor_into(
                &h,
                &options,
                &mut factor,
                &mut LinalgWorkspace::new(),
                execution,
            )
            .unwrap();
            for op in [
                StructuredOperation::Normal,
                StructuredOperation::Transpose,
                StructuredOperation::Adjoint,
            ] {
                let b = rhs(&a, op);
                let x = factor.solve_matrix(op, &b).unwrap();
                check(&a, &b, &x, op, tolerance);
            }
        }
    }
}

#[test]
fn blocked_band_panels_cover_pivots_fill_and_scalar_domains() {
    wide_bands::<f64>();
    wide_bands::<f32>();
    wide_bands::<Complex<f64>>();
    wide_bands::<Complex<f32>>();
}

#[test]
fn long_decaying_structured_solutions_accept_gradual_underflow() {
    for n in [64, 10_000] {
        let off = vec![1.0; n - 1];
        let main = vec![4.0; n];
        let a = TridiagonalMatrixView::new(&off, &main, &off).unwrap();
        let tridiagonal = TridiagonalLuDecomposition::factor(a).unwrap();
        let band = tridiagonal_band(n);
        let lu = BandedLuDecomposition::factor(&band).unwrap();
        let mut lower = vec![1.0; 2 * n];
        for j in 0..n {
            lower[2 * j] = 4.0;
        }
        lower[2 * n - 1] = 0.0;
        let h = HermitianBandMatrix::new(n, 1, Triangle::Lower, lower).unwrap();
        let cholesky = BandedCholeskyDecomposition::factor(&h).unwrap();
        let cyclic = CyclicTridiagonalPlan::new(
            a,
            0.0,
            0.0,
            &scioxide_linalg::LinalgExecutionOptions::new(),
        )
        .unwrap();
        let mut rhs = vec![0.0; n];
        rhs[0] = 1.0;
        let ratio = -1.0 / (2.0 + 3.0_f64.sqrt());
        let expected: Vec<_> = (0..n).map(|i| -ratio.powi(i as i32 + 1)).collect();
        let dense_reference = if n == 64 {
            let mut a = Matrix::zeros(n, n).unwrap();
            for i in 0..n {
                a.view_mut().set(i, i, 4.0).unwrap();
                if i + 1 < n {
                    a.view_mut().set(i, i + 1, 1.0).unwrap();
                    a.view_mut().set(i + 1, i, 1.0).unwrap();
                }
            }
            Some(
                LuDecomposition::factor(&a)
                    .unwrap()
                    .solve(&rhs[..])
                    .unwrap(),
            )
        } else {
            None
        };
        for op in [
            StructuredOperation::Normal,
            StructuredOperation::Transpose,
            StructuredOperation::Adjoint,
        ] {
            for x in [
                tridiagonal.solve(op, &rhs[..]).unwrap(),
                lu.solve(op, &rhs[..]).unwrap(),
                cholesky.solve(op, &rhs[..]).unwrap(),
                cyclic.solve(op, &rhs[..]).unwrap(),
            ] {
                for (i, (&actual, &reference)) in x.as_slice().iter().zip(&expected).enumerate() {
                    assert!(
                        (actual - reference).abs() <= 1e-14 * expected[0],
                        "entry {i}: {actual} vs {reference}"
                    );
                    let residual = 4.0 * actual
                        + if i > 0 { x.as_slice()[i - 1] } else { 0.0 }
                        + if i + 1 < n { x.as_slice()[i + 1] } else { 0.0 }
                        - rhs[i];
                    assert!(residual.abs() < 1e-14);
                }
                if let Some(reference) = &dense_reference {
                    for (&a, &b) in x.as_slice().iter().zip(reference.as_slice()) {
                        assert!((a - b).abs() <= 1e-14 * expected[0]);
                    }
                }
                if n == 10_000 {
                    assert!(x.as_slice().iter().any(|x| x.is_subnormal()));
                }
            }
        }
        // Nonzero corners exercise a correction localized at both ends.
        let cyclic = CyclicTridiagonalPlan::new(
            a,
            1.0,
            1.0,
            &scioxide_linalg::LinalgExecutionOptions::new(),
        )
        .unwrap();
        for op in [
            StructuredOperation::Normal,
            StructuredOperation::Transpose,
            StructuredOperation::Adjoint,
        ] {
            let x = cyclic.solve(op, &rhs[..]).unwrap();
            for (i, &actual) in x.as_slice().iter().enumerate() {
                let reference =
                    (ratio.powi(i as i32) + ratio.powi((n - i) as i32)) / 12.0_f64.sqrt();
                assert!((actual - reference).abs() <= 1e-14 * expected[0]);
            }
        }
    }
}

#[test]
fn cyclic_negative_diagonals_and_singular_auxiliary_factors_match_dense_lu() {
    let mut seed = 123456789_u64;
    let mut next = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (seed >> 11) as f64 / ((1_u64 << 53) as f64) - 0.5
    };
    for n in [3, 4, 17, 65] {
        for case in 0..12 {
            let lower: Vec<_> = (1..n).map(|_| next()).collect();
            let upper: Vec<_> = (1..n).map(|_| next()).collect();
            let main: Vec<_> = (0..n).map(|_| -2.0 - next().abs()).collect();
            let (cu, cl) = (next(), next());
            let (lower, main, upper, cu, cl) = if n == 3 && case == 0 {
                (vec![1.0; 2], vec![-4.0, -4.0, -1.0], vec![1.0; 2], 2.0, 2.0)
            } else if case == 1 {
                // A cyclic shift is invertible; its rank-one auxiliary T is singular.
                (vec![0.0; n - 1], vec![0.0; n], vec![1.0; n - 1], 0.0, 1.0)
            } else {
                (lower, main, upper, cu, cl)
            };
            let tri = TridiagonalMatrixView::new(&lower, &main, &upper).unwrap();
            let plan = CyclicTridiagonalPlan::new(
                tri,
                cu,
                cl,
                &scioxide_linalg::LinalgExecutionOptions::new(),
            )
            .unwrap();
            let mut dense = Matrix::zeros(n, n).unwrap();
            for i in 0..n {
                dense.view_mut().set(i, i, main[i]).unwrap();
                if i + 1 < n {
                    dense.view_mut().set(i, i + 1, upper[i]).unwrap();
                    dense.view_mut().set(i + 1, i, lower[i]).unwrap();
                }
            }
            dense.view_mut().set(0, n - 1, cu).unwrap();
            dense.view_mut().set(n - 1, 0, cl).unwrap();
            let b: Vec<_> = (0..n).map(|_| next()).collect();
            for op in [
                StructuredOperation::Normal,
                StructuredOperation::Transpose,
                StructuredOperation::Adjoint,
            ] {
                let a = if op == StructuredOperation::Normal {
                    dense.view()
                } else {
                    dense.view().transpose()
                };
                let reference = LuDecomposition::factor(a).unwrap().solve(&b[..]).unwrap();
                let actual = plan.solve(op, &b[..]).unwrap();
                let scale = reference
                    .as_slice()
                    .iter()
                    .fold(0.0_f64, |m, x| m.max(x.abs()));
                for (&x, &y) in actual.as_slice().iter().zip(reference.as_slice()) {
                    assert!((x - y).abs() <= 1e-14 * scale);
                }
            }
        }
    }
}

#[test]
fn subnormal_rhs_and_rounded_zero_agree_across_direct_factors() {
    for (diagonal, b, expected) in [
        (1.0, f64::from_bits(1), f64::from_bits(1)),
        (1e308, 1e-308, 0.0),
    ] {
        let dense = Matrix::from_vec(1, 1, vec![diagonal]).unwrap();
        let band = band(&dense, 0, 0);
        let hermitian = HermitianBandMatrix::new(1, 0, Triangle::Lower, vec![diagonal]).unwrap();
        let tridiagonal = TridiagonalLuDecomposition::factor(
            TridiagonalMatrixView::new(&[], &[diagonal], &[]).unwrap(),
        )
        .unwrap();
        let lu = BandedLuDecomposition::factor(&band).unwrap();
        let chol = BandedCholeskyDecomposition::factor(&hermitian).unwrap();
        let triangular =
            TriangularPlan::from_band(band.view(), Triangle::Lower, DiagonalKind::Explicit)
                .unwrap();
        let reference = LuDecomposition::factor(&dense)
            .unwrap()
            .solve(&[b][..])
            .unwrap();
        assert_eq!(reference.as_slice(), &[expected]);
        for op in [
            StructuredOperation::Normal,
            StructuredOperation::Transpose,
            StructuredOperation::Adjoint,
        ] {
            for result in [
                tridiagonal.solve(op, &[b][..]),
                lu.solve(op, &[b][..]),
                chol.solve(op, &[b][..]),
                triangular.solve(op, &[b][..]),
            ] {
                assert_eq!(result.unwrap().as_slice(), reference.as_slice());
            }
        }
    }
}

#[test]
fn wide_band_extreme_scales_and_failed_refactor_preserve_outputs() {
    let (n, k) = (67, 32);
    let options = LinalgExecutionOptions::new();
    for scale in [1e-300, 1e300] {
        let mut a = Matrix::zeros(n, n).unwrap();
        let mut lower = vec![0.0; n * (k + 1)];
        for j in 0..n {
            for i in j..(j + k + 1).min(n) {
                let value = if i == j { scale } else { scale / 128.0 };
                a.view_mut().set(i, j, value).unwrap();
                a.view_mut().set(j, i, value).unwrap();
                lower[j * (k + 1) + i - j] = value;
            }
        }
        let general = band(&a, k, k);
        let hermitian = HermitianBandMatrix::new(n, k, Triangle::Lower, lower).unwrap();
        let mut lu = BandedLuDecomposition::factor(&general).unwrap();
        let mut chol = BandedCholeskyDecomposition::factor(&hermitian).unwrap();
        for op in [
            StructuredOperation::Normal,
            StructuredOperation::Transpose,
            StructuredOperation::Adjoint,
        ] {
            let b = rhs(&a, op);
            let x = lu.solve_matrix(op, &b).unwrap();
            check(&a, &b, &x, op, 1e-14);
            let x = chol.solve_matrix(op, &b).unwrap();
            check(&a, &b, &x, op, 1e-14);
        }
        let before_lu = lu.clone();
        let before_chol = chol.clone();
        let singular =
            BandedMatrix::new(n, n, k, k, 2 * k + 1, vec![0.0; n * (2 * k + 1)]).unwrap();
        let indefinite =
            HermitianBandMatrix::new(n, k, Triangle::Lower, vec![0.0; n * (k + 1)]).unwrap();
        let mut workspace = LinalgWorkspace::new();
        assert!(
            BandedLuDecomposition::factor_into(
                &singular,
                &options,
                &mut lu,
                &mut workspace,
                ExecutionContext::sequential()
            )
            .is_err()
        );
        assert!(
            BandedCholeskyDecomposition::factor_into(
                &indefinite,
                &options,
                &mut chol,
                &mut workspace,
                ExecutionContext::sequential()
            )
            .is_err()
        );
        assert_eq!(lu, before_lu);
        assert_eq!(chol, before_chol);
    }
}

#[test]
fn band_validation_ignores_padding_and_rejects_nonfinite_coefficients_atomically() {
    use scioxide_core::{NonFiniteOrigin, SciOxideErrorView};
    let (n, k) = (35, 16);
    let options = LinalgExecutionOptions::new();
    for execution in [
        ExecutionContext::sequential(),
        ExecutionContext::sequential_native(),
    ] {
        let mut workspace = LinalgWorkspace::new();
        let stride = 2 * k + 3;
        let mut data = vec![f64::NAN; n * stride];
        for j in 0..n {
            for i in j.saturating_sub(k)..(j + k + 1).min(n) {
                data[j * stride + k + i - j] = if i == j { 4.0 } else { 0.01 };
            }
        }
        let mut lu = BandedLuDecomposition::empty();
        BandedLuDecomposition::factor_into(
            BandedMatrixView::new(n, n, k, k, stride, &data).unwrap(),
            &options,
            &mut lu,
            &mut workspace,
            execution,
        )
        .unwrap();
        let before = lu.clone();
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            data[(n / 2) * stride + k + 1] = bad;
            let error = BandedLuDecomposition::factor_into(
                BandedMatrixView::new(n, n, k, k, stride, &data).unwrap(),
                &options,
                &mut lu,
                &mut workspace,
                execution,
            )
            .unwrap_err();
            assert!(matches!(
                error.view(),
                SciOxideErrorView::NonFinite {
                    origin: NonFiniteOrigin::Input,
                    ..
                }
            ));
            assert_eq!(lu, before);
        }
        for triangle in [Triangle::Lower, Triangle::Upper] {
            let mut data = vec![f64::NAN; n * (k + 1)];
            for j in 0..n {
                let rows = match triangle {
                    Triangle::Lower => j..(j + k + 1).min(n),
                    Triangle::Upper => j.saturating_sub(k)..j + 1,
                };
                for i in rows {
                    let slot = match triangle {
                        Triangle::Lower => i - j,
                        Triangle::Upper => k + i - j,
                    };
                    data[j * (k + 1) + slot] = if i == j { 4.0 } else { 0.01 };
                }
            }
            let mut chol = BandedCholeskyDecomposition::empty();
            BandedCholeskyDecomposition::factor_into(
                HermitianBandMatrixView::new(n, k, triangle, &data).unwrap(),
                &options,
                &mut chol,
                &mut workspace,
                execution,
            )
            .unwrap();
            let before = chol.clone();
            for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                let slot = match triangle {
                    Triangle::Lower => 1,
                    Triangle::Upper => k - 1,
                };
                data[(n / 2) * (k + 1) + slot] = bad;
                let error = BandedCholeskyDecomposition::factor_into(
                    HermitianBandMatrixView::new(n, k, triangle, &data).unwrap(),
                    &options,
                    &mut chol,
                    &mut workspace,
                    execution,
                )
                .unwrap_err();
                assert!(matches!(
                    error.view(),
                    SciOxideErrorView::NonFinite {
                        origin: NonFiniteOrigin::Input,
                        ..
                    }
                ));
                assert_eq!(chol, before);
            }
        }
    }
}

#[test]
fn cyclic_band_fallback_preserves_complex_adjoint_semantics() {
    let n = 7;
    let zero = vec![Complex::new(0.0, 0.0); n];
    let upper: Vec<_> = (1..n).map(|i| Complex::new(i as f64, 0.5)).collect();
    let corner = Complex::new(2.0, -0.25);
    let plan = CyclicTridiagonalPlan::new(
        TridiagonalMatrixView::new(&zero[..n - 1], &zero, &upper).unwrap(),
        zero[0],
        corner,
        &scioxide_linalg::LinalgExecutionOptions::new(),
    )
    .unwrap();
    let expected: Vec<_> = (0..n).map(|i| Complex::new(i as f64 / 8.0, -0.5)).collect();
    for op in [
        StructuredOperation::Normal,
        StructuredOperation::Transpose,
        StructuredOperation::Adjoint,
    ] {
        let rhs: Vec<_> = (0..n)
            .map(|i| match op {
                StructuredOperation::Normal => {
                    if i + 1 < n {
                        upper[i] * expected[i + 1]
                    } else {
                        corner * expected[0]
                    }
                }
                _ => {
                    let coefficient = if i == 0 { corner } else { upper[i - 1] };
                    (if op == StructuredOperation::Adjoint {
                        coefficient.conj()
                    } else {
                        coefficient
                    }) * expected[(i + n - 1) % n]
                }
            })
            .collect();
        let actual = plan.solve(op, &rhs[..]).unwrap();
        for (&a, &b) in actual.as_slice().iter().zip(&expected) {
            assert!((a - b).abs() <= 1e-14);
        }
    }
}

#[test]
fn tridiagonal_retries_tiny_residuals_before_small_pivot_division() {
    let tiny = f64::MIN_POSITIVE;
    for op in [
        StructuredOperation::Normal,
        StructuredOperation::Transpose,
        StructuredOperation::Adjoint,
    ] {
        let (main, rhs, expected) = if op == StructuredOperation::Normal {
            ([tiny, 1.0], [0.0, tiny], [-tiny, tiny])
        } else {
            ([1.0, tiny], [tiny, 0.0], [tiny, -tiny])
        };
        let factor = TridiagonalLuDecomposition::factor(
            TridiagonalMatrixView::new(&[0.0], &main, &[tiny]).unwrap(),
        )
        .unwrap();
        let x = factor.solve(op, &rhs[..]).unwrap();
        for (&actual, expected) in x.as_slice().iter().zip(expected) {
            assert!(((actual - expected) / tiny).abs() <= 4.0 * f64::EPSILON);
        }
    }
}

#[test]
fn cholesky_retries_subnormal_schur_diagonals_before_square_root() {
    for (n, k) in [(2, 1), (17, 16), (35, 16)] {
        let root_scale = 1.0_f64.scalbn(-537);
        let mut lower = vec![0.0; n * (k + 1)];
        for j in 0..n {
            lower[j * (k + 1)] = 1.0;
        }
        lower[(n - 2) * (k + 1) + 1] = root_scale / 2.0;
        lower[(n - 1) * (k + 1)] = f64::from_bits(1);
        let a = HermitianBandMatrix::new(n, k, Triangle::Lower, lower).unwrap();
        let factor = BandedCholeskyDecomposition::factor(&a).unwrap();
        let diagonal = factor.lower().unwrap().get(n - 1, n - 1).unwrap();
        assert!((diagonal / root_scale - 0.75_f64.sqrt()).abs() <= 4.0 * f64::EPSILON);
    }
}

#[test]
fn cyclic_plan_matrix_forms_requests_and_options() {
    let tri = TridiagonalMatrix::new(
        vec![-1.0, 0.5, -0.25, 1.0],
        vec![4.0, 5.0, 3.0, 4.5, 6.0],
        vec![1.0, -0.5, 0.75, -1.0],
    )
    .unwrap();
    let options = LinalgExecutionOptions::new();
    let plan = CyclicTridiagonalPlan::new(&tri, -0.5, 0.25, &options).unwrap();
    let rhs = Matrix::from_vec(
        5,
        2,
        vec![1.0, -2.0, 0.5, 3.0, -1.5, 0.25, 2.0, 1.0, 0.75, -0.5],
    )
    .unwrap();
    for op in OPERATIONS {
        let block = plan.solve_matrix(op, &rhs).unwrap();
        for j in 0..2 {
            let column = plan.solve(op, rhs.view().col(j).unwrap()).unwrap();
            for i in 0..5 {
                assert_eq!(block.view().get(i, j).unwrap(), column.as_slice()[i]);
            }
        }
    }
    // Real adjoint solves are transposed solves.
    assert_eq!(
        plan.solve_matrix(StructuredOperation::Adjoint, &rhs)
            .unwrap(),
        plan.solve_matrix(StructuredOperation::Transpose, &rhs)
            .unwrap()
    );
    let requirements = |request: StructuredWorkspaceRequest| plan.workspace_requirements(&request);
    assert_eq!(
        requirements(StructuredWorkspaceRequest::solve_matrix(5, 2, &options))
            .unwrap()
            .scalar_len()
            .unwrap(),
        30
    );
    assert_eq!(
        requirements(StructuredWorkspaceRequest::factor(5, 1, 1, &options))
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::InvalidInput
    );
    assert_eq!(
        requirements(StructuredWorkspaceRequest::solve(4, &options))
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(
        requirements(StructuredWorkspaceRequest::solve(
            5,
            &options.with_byte_limit(8)
        ))
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::ResourceLimit
    );
    // Preparation runs under the options' limits and iteration budget.
    assert_eq!(
        CyclicTridiagonalPlan::new(&tri, -0.5, 0.25, &options.with_byte_limit(8))
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::ResourceLimit
    );
    assert_eq!(
        CyclicTridiagonalPlan::build(
            &tri,
            -0.5,
            0.25,
            &options.with_iteration_budget(Budget::ONE),
            ExecutionContext::sequential(),
        )
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::ConvergenceFailure
    );
    assert_eq!(
        TridiagonalLuDecomposition::<f64>::workspace_requirements(
            &StructuredWorkspaceRequest::factor(5, 2, 1, &options)
        )
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::InvalidInput
    );
}

#[test]
fn cyclic_policy_rejection_and_execution_failures_preserve_outputs() {
    let execution = LinalgExecutionOptions::new();
    let matrix = TridiagonalMatrixView::new(&[0.0; 2], &[1.0, 1e-12, 2.0], &[0.0; 2]).unwrap();
    let plan = CyclicTridiagonalPlan::new(matrix, 0.0, 0.0, &execution).unwrap();
    let exact = DirectSolveOptions::new(RankPolicy::Exact);
    let threshold = RankPolicy::Threshold(scioxide_core::RankThreshold::new(1e-8, 0.0).unwrap());
    assert_eq!(plan.rank(threshold).unwrap().rank(), 2);
    assert_eq!(plan.rank(RankPolicy::Exact).unwrap().rank(), 3);
    let mut workspace = plan
        .workspace(&StructuredWorkspaceRequest::solve_matrix(3, 2, &execution))
        .unwrap();
    let b = Matrix::from_vec(3, 2, vec![1.0; 6]).unwrap();
    for (options, termination) in [
        (
            DirectSolveOptions::new(threshold),
            DirectSolveTermination::RankDeficient,
        ),
        (
            exact
                .with_minimum_reciprocal_condition(ReciprocalConditionThreshold::new(0.1).unwrap()),
            DirectSolveTermination::IllConditioned,
        ),
    ] {
        for op in OPERATIONS {
            let mut vector = [77.0; 3];
            let report = plan
                .solve_into(
                    op,
                    &[1.0; 3][..],
                    &options,
                    VectorViewMut::from_slice(&mut vector),
                    &mut workspace,
                    ExecutionContext::sequential_native(),
                )
                .unwrap();
            assert_eq!(report.termination(), termination);
            assert_eq!(vector, [77.0; 3]);
            let mut output = Matrix::from_vec(3, 2, vec![77.0; 6]).unwrap();
            let report = plan
                .solve_matrix_into(
                    op,
                    &b,
                    &options,
                    output.view_mut(),
                    &mut workspace,
                    ExecutionContext::sequential_native(),
                )
                .unwrap();
            assert_eq!(report.termination(), termination);
            assert!(output.as_slice().iter().all(|&x| x == 77.0));
            let attempt = workspace.last_attempt().unwrap();
            assert_eq!(attempt.outcome(), AttemptOutcome::Succeeded);
            assert_eq!(attempt.counts().own().iterations(), 0);
            assert!(
                plan.solve_with_options(op, &[1.0; 3][..], &options)
                    .unwrap()
                    .solution()
                    .is_none()
            );
            assert!(
                plan.solve_matrix_with_options(op, &b, &options)
                    .unwrap()
                    .solution()
                    .is_none()
            );
        }
    }
    let mut output = Matrix::from_vec(3, 2, vec![77.0; 6]).unwrap();
    let limited = exact.with_execution_options(execution.with_iteration_budget(Budget::ONE));
    assert_eq!(
        plan.solve_matrix_into(
            StructuredOperation::Normal,
            &b,
            &limited,
            output.view_mut(),
            &mut workspace,
            ExecutionContext::sequential_native()
        )
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::ConvergenceFailure
    );
    assert!(output.as_slice().iter().all(|&x| x == 77.0));
    assert_eq!(
        workspace.last_attempt().unwrap().outcome(),
        AttemptOutcome::Stopped
    );
    let huge = Matrix::from_vec(3, 2, vec![f64::MAX; 6]).unwrap();
    assert!(
        plan.solve_matrix_into(
            StructuredOperation::Normal,
            &huge,
            &exact,
            output.view_mut(),
            &mut workspace,
            ExecutionContext::sequential_native()
        )
        .is_err()
    );
    assert!(output.as_slice().iter().all(|&x| x == 77.0));
    assert_eq!(
        workspace.last_attempt().unwrap().outcome(),
        AttemptOutcome::Failed
    );
    let token = scioxide_core::CancellationToken::new();
    token.cancel();
    assert_eq!(
        plan.solve_matrix_into(
            StructuredOperation::Normal,
            &b,
            &exact,
            output.view_mut(),
            &mut workspace,
            ExecutionContext::sequential().with_cancellation(&token)
        )
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::ExecutionStopped
    );
    assert!(output.as_slice().iter().all(|&x| x == 77.0));
    let limited = exact.with_execution_options(execution.with_byte_limit(8));
    assert_eq!(
        plan.solve_matrix_into(
            StructuredOperation::Normal,
            &b,
            &limited,
            output.view_mut(),
            &mut workspace,
            ExecutionContext::sequential_native()
        )
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::ResourceLimit
    );
    let empty = Matrix::zeros(3, 0).unwrap();
    let result = plan
        .solve_matrix_with_options(
            StructuredOperation::Normal,
            &empty,
            &exact.with_residual(Estimation::Estimate),
        )
        .unwrap();
    assert_eq!(result.solution().unwrap().cols(), 0);
    assert_eq!(result.report().rank().rank(), 3);
    assert!(result.report().condition().is_none() && result.report().residual().is_none());
}

#[test]
fn complex_cyclic_adjoint_derives_its_own_correction() {
    let c = |re: f64, im: f64| Complex::new(re, im);
    let tri = TridiagonalMatrix::new(
        vec![c(-1.0, 0.5), c(0.5, -1.0), c(0.25, 0.0)],
        vec![c(4.0, 1.0), c(5.0, -0.5), c(3.0, 2.0), c(4.0, 0.0)],
        vec![c(1.0, 1.0), c(-0.5, 0.0), c(0.75, -0.25)],
    )
    .unwrap();
    let (upper, lower) = (c(-0.5, 0.25), c(0.25, -1.0));
    let plan =
        CyclicTridiagonalPlan::new(&tri, upper, lower, &LinalgExecutionOptions::new()).unwrap();
    // Dense cyclic matrix for the reference residual.
    let dense = |i: usize, j: usize| match (i, j) {
        (0, 3) => upper,
        (3, 0) => lower,
        _ if i == j => tri.main()[i],
        _ if i == j + 1 => tri.lower()[j],
        _ if j == i + 1 => tri.upper()[i],
        _ => c(0.0, 0.0),
    };
    let rhs = [c(1.0, -1.0), c(0.5, 2.0), c(-1.0, 0.0), c(2.0, 0.5)];
    for op in OPERATIONS {
        let x = plan.solve(op, &rhs[..]).unwrap();
        for i in 0..4 {
            let value = (0..4).fold(c(0.0, 0.0), |sum, j| {
                let a = match op {
                    StructuredOperation::Normal => dense(i, j),
                    StructuredOperation::Transpose => dense(j, i),
                    StructuredOperation::Adjoint => dense(j, i).conj(),
                };
                sum + a * x.as_slice()[j]
            });
            assert!((value - rhs[i]).norm() < 1e-13);
        }
    }
}

#[test]
fn compact_tridiagonal_execution_checks_interrupt_both_pivot_and_dominant_paths() {
    use scioxide_linalg::backend::tridiagonal::TridiagonalFactor;
    use std::sync::atomic::{AtomicUsize, Ordering};
    struct StopAfter(AtomicUsize);
    impl scioxide_core::Cancellation for StopAfter {
        fn is_cancelled(&self) -> bool {
            self.0.fetch_add(1, Ordering::Relaxed) >= 2
        }
    }
    for off in [-1.0_f64, 1.0] {
        let cancel = StopAfter(AtomicUsize::new(0));
        let error = TridiagonalFactor::factor(
            vec![off; 1023],
            vec![8.0; 1024],
            vec![off; 1023],
            ExecutionContext::sequential().with_cancellation(&cancel),
        )
        .unwrap_err();
        assert_eq!(error.kind(), SciOxideErrorKind::ExecutionStopped);
        let factor = TridiagonalFactor::factor(
            vec![off; 1023],
            vec![8.0; 1024],
            vec![off; 1023],
            scioxide_core::ExecutionContext::sequential_native(),
        )
        .unwrap();
        let cancel = StopAfter(AtomicUsize::new(0));
        let error = factor
            .solve_in_place(
                &mut [1.0; 1024],
                ExecutionContext::sequential().with_cancellation(&cancel),
            )
            .unwrap_err();
        assert_eq!(error.kind(), SciOxideErrorKind::ExecutionStopped);
    }
}

#[test]
fn compact_refactoring_reuses_storage_and_recovers_after_failure() {
    use scioxide_linalg::backend::tridiagonal::TridiagonalFactor;
    let execution = ExecutionContext::sequential();
    let mut factor = TridiagonalFactor::factor(
        vec![],
        vec![1.0_f64],
        vec![],
        scioxide_core::ExecutionContext::sequential_native(),
    )
    .unwrap();
    let lower = [2.0, 1.0];
    let main = [0.0, 3.0, 4.0];
    let upper = [1.0, 2.0];
    factor.refactor(&lower, &main, &upper, execution).unwrap();
    factor.refactor(&lower, &main, &upper, execution).unwrap();
    let mut x = [2.0, 14.0, 14.0];
    factor
        .solve_in_place(&mut x, scioxide_core::ExecutionContext::sequential_native())
        .unwrap();
    for (actual, expected) in x.into_iter().zip([1.0, 2.0, 3.0]) {
        assert!((actual - expected).abs() < 1e-12);
    }
    assert!(
        factor
            .refactor(&[0.0; 2], &[0.0; 3], &[0.0; 2], execution)
            .is_err()
    );
    assert!(
        factor
            .solve_in_place(
                &mut [1.0; 3],
                scioxide_core::ExecutionContext::sequential_native()
            )
            .is_err()
    );
    factor.refactor(&lower, &main, &upper, execution).unwrap();
    let mut x = [2.0, 14.0, 14.0];
    factor
        .solve_in_place(&mut x, scioxide_core::ExecutionContext::sequential_native())
        .unwrap();
    assert!((x[2] - 3.0).abs() < 1e-12);
    let token = scioxide_core::CancellationToken::new();
    token.cancel();
    assert_eq!(
        factor
            .refactor(&lower, &main, &upper, execution.with_cancellation(&token))
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::ExecutionStopped
    );
}

/// Dense reference solve of an `n × n` row-major system with the LU factorization.
fn dense_reference(n: usize, a: &[f64], b: &[f64]) -> Vec<f64> {
    LuDecomposition::factor(&Matrix::from_vec(n, n, a.to_vec()).unwrap())
        .unwrap()
        .solve(b)
        .unwrap()
        .as_slice()
        .to_vec()
}

fn assert_relative(actual: &[f64], expected: &[f64], tolerance: f64) {
    let scale = expected.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    for (x, y) in actual.iter().zip(expected) {
        assert!(
            (x - y).abs() <= tolerance * scale,
            "{actual:?} != {expected:?}"
        );
    }
}

#[test]
fn cyclic_solves_are_accurate_when_the_rank_one_split_is_nearly_singular() {
    // rcond₁(A) ≈ 0.11, while the tridiagonal part of a Sherman–Morrison split
    // with γ = −a₀₀ is nearly singular.
    let m3 = 4.0 / 87.0;
    let a = [
        1.0, 1.0, 0.0, 1.0, //
        1.0, 3.0, 1.0, 0.0, //
        0.0, 1.0, 3.0, 1.0, //
        1.0, 0.0, 1.0, m3,
    ];
    let transposed: Vec<f64> = (0..16).map(|k| a[(k % 4) * 4 + k / 4]).collect();
    let b = [1.0, 2.0, 3.0, 4.0];
    let plan = CyclicTridiagonalPlan::new(
        TridiagonalMatrixView::new(&[1.0; 3], &[1.0, 3.0, 3.0, m3], &[1.0; 3]).unwrap(),
        1.0,
        1.0,
        &LinalgExecutionOptions::new(),
    )
    .unwrap();
    let options = DirectSolveOptions::default().with_residual(Estimation::Estimate);
    for (op, matrix) in [
        (StructuredOperation::Normal, &a[..]),
        (StructuredOperation::Transpose, &transposed[..]),
    ] {
        let result = plan.solve_with_options(op, &b[..], &options).unwrap();
        assert_eq!(
            result.report().termination(),
            DirectSolveTermination::Accepted
        );
        assert!(result.report().residual().unwrap().scaled_norm() < 1e-15);
        assert_relative(
            result.solution().unwrap().as_slice(),
            &dense_reference(4, matrix, &b),
            1e-14,
        );
    }
    // The backend factor: with γ = −a₀₀ its split T is singular at a₃₃ = −8/13.
    let m3 = -8.0 / 13.0 + 1e-15;
    let a = [
        1.0, 1.0, 0.0, 1.0, //
        1.0, 3.0, 1.0, 0.0, //
        0.0, 1.0, 3.0, 1.0, //
        1.0, 0.0, 1.0, m3,
    ];
    let factor = scioxide_linalg::backend::tridiagonal::CyclicTridiagonalFactor::factor(
        vec![1.0; 3],
        vec![1.0, 3.0, 3.0, m3],
        vec![1.0; 3],
        1.0,
        1.0,
    )
    .unwrap();
    let mut x = b;
    factor.solve_in_place(&mut x).unwrap();
    assert_relative(&x, &dense_reference(4, &a, &b), 1e-14);
}

#[test]
fn band_factors_allow_gradual_underflow_of_multipliers() {
    // A = [1e300 1; 1e-300 2] is well conditioned after scaling; its
    // multiplier 1e-600 underflows to zero. A [1, 1] = [1e300 + 1, 2].
    let b = [1e300, 2.0];
    let band = BandedMatrix::new(2, 2, 1, 1, 3, vec![0.0, 1e300, 1e-300, 1.0, 2.0, 0.0]).unwrap();
    let x = BandedLuDecomposition::factor(&band)
        .unwrap()
        .solve(StructuredOperation::Normal, &b[..])
        .unwrap();
    assert_relative(x.as_slice(), &[1.0, 1.0], 1e-15);
    let tridiagonal = TridiagonalMatrix::new(vec![1e-300], vec![1e300, 2.0], vec![1.0]).unwrap();
    let x = TridiagonalLuDecomposition::factor(&tridiagonal)
        .unwrap()
        .solve(StructuredOperation::Normal, &b[..])
        .unwrap();
    assert_relative(x.as_slice(), &[1.0, 1.0], 1e-15);
    let mut rhs = b;
    scioxide_linalg::backend::tridiagonal::TridiagonalFactor::factor(
        vec![1e-300],
        vec![1e300, 2.0],
        vec![1.0],
        ExecutionContext::sequential_native(),
    )
    .unwrap()
    .solve_in_place(&mut rhs, ExecutionContext::sequential_native())
    .unwrap();
    assert_relative(&rhs, &[1.0, 1.0], 1e-15);
    // The cyclic plan of [1e300 1 1; 1e-300 2 1; 1 1 2] maps [1, 1, 1] to b.
    let plan = CyclicTridiagonalPlan::new(
        TridiagonalMatrixView::new(&[1e-300, 1.0], &[1e300, 2.0, 2.0], &[1.0, 1.0]).unwrap(),
        1.0,
        1.0,
        &LinalgExecutionOptions::new(),
    )
    .unwrap();
    let x = plan
        .solve(
            StructuredOperation::Normal,
            &[1e300 + 2.0, 1e-300 + 3.0, 4.0][..],
        )
        .unwrap();
    assert_relative(x.as_slice(), &[1.0, 1.0, 1.0], 1e-15);
}

#[test]
fn reserved_dense_fallback_covers_more_columns_than_requested() {
    // An f32 Toeplitz system whose GKO solution misses a 6e-8 tolerance.
    let column: Vec<f32> = (0..9).map(|k| ((k * 7 % 5) as f32) - 2.0).collect();
    let mut row: Vec<f32> = (0..9).map(|k| ((k * 3 % 7) as f32) - 3.0).collect();
    row[0] = column[0];
    let options = StructuredSolveOptions::<f32>::new()
        .with_tolerance(scioxide_core::Tolerance::new(0.0, 6e-8).unwrap());
    let plan = StructuredSolvePlan::new(
        StructuredMatrixView::toeplitz(&column, &row).unwrap(),
        &options,
    )
    .unwrap();
    let request =
        StructuredWorkspaceRequest::solve(9, &LinalgExecutionOptions::new()).with_dense_fallback();
    let mut workspace = plan.workspace(&request).unwrap();
    let rhs: Vec<f32> = (0..18).map(|k| (k / 2 % 9) as f32 + 1.0).collect();
    let rhs = Matrix::from_vec(9, 2, rhs).unwrap();
    let mut output = Matrix::zeros(9, 2).unwrap();
    let report = plan
        .solve_matrix_into(
            StructuredOperation::Normal,
            &rhs,
            output.view_mut(),
            &mut workspace,
            ExecutionContext::sequential_native(),
        )
        .unwrap();
    assert_eq!(report.termination(), DirectSolveTermination::Accepted);
    assert_eq!(report.dense_lu_solves(), 2);
    assert!(report.backward_error() <= 6e-8);
}
