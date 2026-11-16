//! Independent reconstruction, orthogonality and edit tests for orthogonal factors.
#![allow(clippy::unwrap_used)]
use scioxide_core::{Complex, NumericScalar, RealScalar};
use scioxide_linalg::{
    LinalgWorkspace, Matrix, QrModification, QrUpdateDecomposition, QrUpdateOptions, RankPolicy,
    RqDecomposition,
};
fn real<S: NumericScalar>(x: f64) -> S {
    S::from_real(S::Real::try_from_f64_lossy(x).unwrap())
}
fn matrix<S: NumericScalar>(m: usize, n: usize, seed: usize) -> Matrix<S> {
    Matrix::from_vec(
        m,
        n,
        (0..m * n)
            .map(|i| {
                let re = ((i * 17 + seed * 3) % 29) as f64 / 13.0 - 0.7;
                let im = if core::mem::size_of::<S>() > core::mem::size_of::<S::Real>() {
                    ((i * 7 + seed) % 13) as f64 / 11.0 - 0.3
                } else {
                    0.0
                };
                S::from_components(
                    S::Real::try_from_f64_lossy(re).unwrap(),
                    S::Real::try_from_f64_lossy(im).unwrap(),
                )
            })
            .collect(),
    )
    .unwrap()
}
fn options<S: NumericScalar>() -> QrUpdateOptions<S::Real> {
    QrUpdateOptions::new(RankPolicy::DimensionScaledEpsilon)
}
// Scalar triple loop independent of SciOxide's factor and GEMM kernels.
fn multiply<S: NumericScalar>(a: &Matrix<S>, b: &Matrix<S>) -> Matrix<S> {
    Matrix::from_vec(
        a.rows(),
        b.cols(),
        (0..a.rows() * b.cols())
            .map(|ij| {
                (0..a.cols()).fold(S::ZERO, |s, k| {
                    s + a.view().get(ij / b.cols(), k).unwrap()
                        * b.view().get(k, ij % b.cols()).unwrap()
                })
            })
            .collect(),
    )
    .unwrap()
}
fn close<S: NumericScalar>(a: &Matrix<S>, b: &Matrix<S>, steps: usize) {
    assert_eq!((a.rows(), a.cols()), (b.rows(), b.cols()));
    let scale = b
        .as_slice()
        .iter()
        .fold(0.0_f64, |s, x| s.hypot(x.abs().into_f64()))
        .max(1.0);
    let error = crate::support::comparisons::matrix_error(a, b) / scale;
    let tol = 32.0
        * S::Real::EPSILON.into_f64()
        * a.rows().max(a.cols()).max(1) as f64
        * (steps.max(1) as f64).sqrt();
    assert!(
        error <= tol,
        "error {error} > {tol}, {}x{}",
        a.rows(),
        a.cols()
    );
}
fn unitary<S: NumericScalar>(q: &Matrix<S>, steps: usize) {
    close(
        &multiply(&q.view().adjoint().try_to_matrix().unwrap(), q),
        &Matrix::identity(q.cols()).unwrap(),
        steps,
    );
}
fn verify<S: NumericScalar>(factor: &QrUpdateDecomposition<S>, a: &Matrix<S>, steps: usize) {
    let qr = multiply(factor.q_matrix(), factor.r_matrix());
    let permuted = Matrix::from_vec(
        a.rows(),
        a.cols(),
        (0..a.rows() * a.cols())
            .map(|ij| {
                a.view()
                    .get(ij / a.cols(), factor.permutation()[ij % a.cols()])
                    .unwrap()
            })
            .collect(),
    )
    .unwrap();
    close(&qr, &permuted, steps);
    unitary(factor.q_matrix(), steps);
    for i in 0..a.rows() {
        for j in 0..i.min(a.cols()) {
            assert_eq!(factor.r_matrix().view().get(i, j).unwrap(), S::ZERO);
        }
    }
}
fn edited<S: NumericScalar>(a: &Matrix<S>, edit: QrModification<'_, S>) -> Matrix<S> {
    let (m, n) = (a.rows(), a.cols());
    let (rows, cols) = match edit {
        QrModification::InsertRows { values, .. } => (m + values.rows(), n),
        QrModification::DeleteRows { count, .. } => (m - count, n),
        QrModification::InsertColumns { values, .. } => (m, n + values.cols()),
        QrModification::DeleteColumns { count, .. } => (m, n - count),
        _ => (m, n),
    };
    Matrix::from_vec(
        rows,
        cols,
        (0..rows * cols)
            .map(|ij| {
                let (i, j) = (ij / cols, ij % cols);
                match edit {
                    QrModification::InsertRows { index, values } => {
                        if i < index {
                            a.view().get(i, j).unwrap()
                        } else if i < index + values.rows() {
                            values.get(i - index, j).unwrap()
                        } else {
                            a.view().get(i - values.rows(), j).unwrap()
                        }
                    }
                    QrModification::DeleteRows { index, count } => a
                        .view()
                        .get(if i < index { i } else { i + count }, j)
                        .unwrap(),
                    QrModification::InsertColumns { index, values } => {
                        if j < index {
                            a.view().get(i, j).unwrap()
                        } else if j < index + values.cols() {
                            values.get(i, j - index).unwrap()
                        } else {
                            a.view().get(i, j - values.cols()).unwrap()
                        }
                    }
                    QrModification::DeleteColumns { index, count } => a
                        .view()
                        .get(i, if j < index { j } else { j + count })
                        .unwrap(),
                    QrModification::RankUpdate { u, v } | QrModification::RankDowndate { u, v } => {
                        let change = (0..u.cols()).fold(S::ZERO, |s, k| {
                            s + u.get(i, k).unwrap() * v.get(j, k).unwrap().conj()
                        });
                        a.view().get(i, j).unwrap()
                            + if matches!(edit, QrModification::RankDowndate { .. }) {
                                -change
                            } else {
                                change
                            }
                    }
                }
            })
            .collect(),
    )
    .unwrap()
}
fn cases<S: NumericScalar>() {
    for (m, n) in [(4, 4), (7, 3), (3, 7), (0, 3), (3, 0), (0, 0)] {
        let a = matrix::<S>(m, n, 2);
        let rq = RqDecomposition::factor(a.view()).unwrap();
        close(&multiply(rq.r_matrix(), rq.q_matrix()), &a, 1);
        unitary(rq.q_matrix(), 1);
        for i in 0..m {
            for j in 0..n {
                if i + n > j + m {
                    assert_eq!(rq.r_matrix().view().get(i, j).unwrap(), S::ZERO);
                }
            }
        }
        let factor =
            QrUpdateDecomposition::factor_with_options(a.view(), &(options::<S>())).unwrap();
        verify(&factor, &a, 1);
        let mut work = LinalgWorkspace::new();
        let rows = matrix::<S>(2, n, 4);
        let cols = matrix::<S>(m, 2, 5);
        for index in 0..=m {
            let edit = QrModification::InsertRows {
                index,
                values: rows.view(),
            };
            let target = edited(&a, edit);
            let mut f = factor.clone();
            f.modify_in_place(
                edit,
                &(options::<S>()),
                &mut work,
                scioxide_core::ExecutionContext::sequential(),
            )
            .unwrap();
            verify(&f, &target, 1);
            f.modify_in_place(
                QrModification::DeleteRows { index, count: 2 },
                &(options::<S>()),
                &mut work,
                scioxide_core::ExecutionContext::sequential(),
            )
            .unwrap();
            verify(&f, &a, 2);
        }
        for index in 0..=n {
            let edit = QrModification::InsertColumns {
                index,
                values: cols.view(),
            };
            let target = edited(&a, edit);
            let mut f = factor.clone();
            f.modify_in_place(
                edit,
                &(options::<S>()),
                &mut work,
                scioxide_core::ExecutionContext::sequential(),
            )
            .unwrap();
            verify(&f, &target, 1);
            f.modify_in_place(
                QrModification::DeleteColumns { index, count: 2 },
                &(options::<S>()),
                &mut work,
                scioxide_core::ExecutionContext::sequential(),
            )
            .unwrap();
            verify(&f, &a, 2);
        }
        for index in 0..m {
            let edit = QrModification::DeleteRows { index, count: 1 };
            let mut f = factor.clone();
            f.modify_in_place(
                edit,
                &(options::<S>()),
                &mut work,
                scioxide_core::ExecutionContext::sequential(),
            )
            .unwrap();
            verify(&f, &edited(&a, edit), 1);
        }
        for index in 0..n {
            let edit = QrModification::DeleteColumns { index, count: 1 };
            let mut f = factor.clone();
            f.modify_in_place(
                edit,
                &(options::<S>()),
                &mut work,
                scioxide_core::ExecutionContext::sequential(),
            )
            .unwrap();
            verify(&f, &edited(&a, edit), 1);
        }
        for k in [0, 1, 3] {
            let u = matrix::<S>(m, k, 9);
            let v = matrix::<S>(n, k, 7);
            let mut f = factor.clone();
            let edit = QrModification::RankUpdate {
                u: u.view(),
                v: v.view(),
            };
            f.modify_in_place(
                edit,
                &(options::<S>()),
                &mut work,
                scioxide_core::ExecutionContext::sequential(),
            )
            .unwrap();
            verify(&f, &edited(&a, edit), 1);
            f.modify_in_place(
                QrModification::RankDowndate {
                    u: u.view(),
                    v: v.view(),
                },
                &(options::<S>()),
                &mut work,
                scioxide_core::ExecutionContext::sequential(),
            )
            .unwrap();
            verify(&f, &a, 2);
        }
    }
}
#[test]
fn real32() {
    cases::<f32>();
}
#[test]
fn real64() {
    cases::<f64>();
}
#[test]
fn complex32() {
    cases::<Complex<f32>>();
}
#[test]
fn complex64() {
    cases::<Complex<f64>>();
}
fn repeated<S: NumericScalar>() {
    let a = matrix::<S>(7, 4, 2);
    let mut f = QrUpdateDecomposition::factor_with_options(a.view(), &(options::<S>())).unwrap();
    let mut work = LinalgWorkspace::new();
    let u = matrix::<S>(7, 2, 8);
    let v = matrix::<S>(4, 2, 9);
    for step in 0..40 {
        let edit = QrModification::RankUpdate {
            u: u.view(),
            v: v.view(),
        };
        f.modify_in_place(
            edit,
            &(options::<S>()),
            &mut work,
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap();
        verify(&f, &edited(&a, edit), 2 * step + 1);
        f.modify_in_place(
            QrModification::RankDowndate {
                u: u.view(),
                v: v.view(),
            },
            &(options::<S>()),
            &mut work,
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap();
        verify(&f, &a, 2 * step + 2);
    }
    let a = Matrix::<S>::identity(3).unwrap();
    let mut f = QrUpdateDecomposition::factor_with_options(a.view(), &(options::<S>())).unwrap();
    let u = Matrix::from_vec(3, 1, vec![S::ONE, S::ZERO, S::ZERO]).unwrap();
    let edit = QrModification::RankDowndate {
        u: u.view(),
        v: u.view(),
    };
    let report = f
        .modify_in_place(
            edit,
            &(options::<S>()),
            &mut work,
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap();
    assert!(report.rank_lost());
    assert_eq!(report.after().report().rank(), 2);
    verify(&f, &edited(&a, edit), 1);
    let report = f
        .modify_in_place(
            QrModification::RankUpdate {
                u: u.view(),
                v: u.view(),
            },
            &(options::<S>()),
            &mut work,
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(report.after().report().rank(), 3);
    verify(&f, &a, 2);
    // A zero diagonal followed by a nonzero off-diagonal still has rank one.
    let a = Matrix::from_vec(2, 2, vec![S::ZERO, S::ONE, S::ZERO, S::ZERO]).unwrap();
    let f = QrUpdateDecomposition::factor_with_options(a.view(), &(options::<S>())).unwrap();
    assert_eq!(f.rank_decision().report().rank(), 1);
}
#[test]
fn repeated_and_rank() {
    repeated::<f32>();
    repeated::<f64>();
    repeated::<Complex<f32>>();
    repeated::<Complex<f64>>();
}

#[test]
fn failed_modifications_are_transactional_and_retryable() {
    use scioxide_core::{
        Budget, BudgetKind, CancellationToken, ExecutionContext, ProblemToken, SolveBudgets,
        SolveIdentity, SolveState,
    };
    let a = matrix::<f64>(4, 3, 7);
    let mut f = QrUpdateDecomposition::factor_with_options(a.view(), &(options::<f64>())).unwrap();
    let saved = f.clone();
    let mut w = LinalgWorkspace::new();
    let bad = Matrix::from_vec(4, 1, vec![f64::NAN; 4]).unwrap();
    let v = matrix::<f64>(3, 1, 8);
    assert!(matches!(
        (f.modify_in_place(
            QrModification::RankDowndate {
                u: bad.view(),
                v: v.view()
            },
            &(options::<f64>()),
            &mut w,
            scioxide_core::ExecutionContext::sequential()
        ))
        .as_ref()
        .err()
        .unwrap()
        .view(),
        scioxide_core::SciOxideErrorView::NonFinite { .. }
    ));
    assert!(
        f.modify_in_place(
            QrModification::DeleteRows { index: 3, count: 2 },
            &(options::<f64>()),
            &mut w,
            scioxide_core::ExecutionContext::sequential()
        )
        .is_err()
    );
    let rows = matrix::<f64>(2, 3, 8);
    let edit = QrModification::InsertRows {
        index: 1,
        values: rows.view(),
    };
    let token = CancellationToken::new();
    token.cancel();
    assert!(
        f.modify_in_place(
            edit,
            &(options::<f64>()),
            &mut w,
            ExecutionContext::sequential().with_cancellation(&token)
        )
        .is_err()
    );
    let mut budgets =
        SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::try_from(1).unwrap());
    let mut state = SolveState::new();
    let id = SolveIdentity::new(
        ProblemToken::new(664),
        scioxide_core::algorithm_id!("linalg", "qr_update", 1),
        4,
        0,
    );
    let mut solve = state
        .begin_solve(ExecutionContext::sequential(), id, &mut budgets)
        .unwrap();
    assert!(matches!(
        (f.modify_in_place(
            edit,
            &options::<f64>().with_execution_options(
                scioxide_linalg::LinalgExecutionOptions::new().with_iteration_budget(Budget::ONE)
            ),
            &mut w,
            &mut solve
        ))
        .as_ref()
        .err()
        .unwrap()
        .view(),
        scioxide_core::SciOxideErrorView::ConvergenceFailure { .. }
    ));
    assert_eq!(solve.counts().own().iterations(), 0);
    assert!(solve.counts().nested().iterations() > 0);
    let attempt = w.last_attempt().unwrap();
    assert_eq!(attempt.outcome(), scioxide_core::AttemptOutcome::Stopped);
    assert_eq!(attempt.counts().own().iterations(), 1);
    assert_eq!(f.q_matrix(), saved.q_matrix());
    assert_eq!(f.r_matrix(), saved.r_matrix());
    assert_eq!(f.permutation(), saved.permutation());
    assert_eq!(f.rank_decision(), saved.rank_decision());
    f.modify_in_place(
        edit,
        &(options::<f64>()),
        &mut w,
        scioxide_core::ExecutionContext::sequential(),
    )
    .unwrap();
    verify(&f, &edited(&a, edit), 1);
}
#[test]
fn borrowed_transposed_views_and_existing_householder() {
    use scioxide_linalg::{HouseholderDecomposition, HouseholderOrientation};
    let a = matrix::<Complex<f64>>(4, 7, 6);
    let transposed = a.view().transpose().try_to_matrix().unwrap();
    let rq = RqDecomposition::factor(transposed.view().transpose()).unwrap();
    close(&multiply(rq.r_matrix(), rq.q_matrix()), &a, 1);
    let packed = HouseholderDecomposition::factor(
        &a,
        HouseholderOrientation::Qr,
        scioxide_linalg::HouseholderPivoting::Column,
    )
    .unwrap();
    let f =
        QrUpdateDecomposition::from_householder_with_options(&packed, &(options::<Complex<f64>>()))
            .unwrap();
    verify(&f, &a, 1);
    let lq = HouseholderDecomposition::factor(
        &a,
        HouseholderOrientation::Lq,
        scioxide_linalg::HouseholderPivoting::Column,
    )
    .unwrap();
    let error =
        QrUpdateDecomposition::from_householder_with_options(&lq, &(options::<Complex<f64>>()))
            .unwrap_err();
    assert!(matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::InvalidInput {
            reason: scioxide_core::InvalidInputReason::UnsupportedConfiguration,
            ..
        }
    ));
    let f = QrUpdateDecomposition::factor_with_options(
        transposed.view().transpose(),
        &(options::<Complex<f64>>()),
    )
    .unwrap();
    verify(&f, &a, 1);
}
#[test]
#[cfg(feature = "std")]
fn executor_matches_sequential() {
    use scioxide_core::{ExecutionContext, ExecutionPolicy};
    use scioxide_linalg::{LinalgExecutionOptions, LinalgWorkspace};
    let workers = core::num::NonZeroUsize::new(2).unwrap();
    let pool = scioxide_core::ThreadExecutor::new(workers);
    let context = ExecutionContext::new(&pool);
    let policy = ExecutionPolicy::Parallel(workers);
    let a = matrix::<Complex<f64>>(8, 5, 4);
    let opts = options::<Complex<f64>>()
        .with_execution_options(LinalgExecutionOptions::new().with_execution_policy(policy));
    let mut f = QrUpdateDecomposition::empty();
    let mut w = LinalgWorkspace::new();
    w.try_reserve(
        &QrUpdateDecomposition::<Complex<f64>>::workspace_requirements(
            &scioxide_linalg::QrUpdateWorkspaceRequest::new(8, 5, &opts),
        )
        .unwrap(),
    )
    .unwrap();
    QrUpdateDecomposition::factor_into(a.view(), &(opts), &mut f, &mut w, context).unwrap();
    verify(&f, &a, 1);
    let u = matrix::<Complex<f64>>(8, 2, 4);
    let v = matrix::<Complex<f64>>(5, 2, 7);
    let edit = QrModification::RankUpdate {
        u: u.view(),
        v: v.view(),
    };
    f.modify_in_place(edit, &(opts), &mut w, context).unwrap();
    verify(&f, &edited(&a, edit), 1);
    let mut rq = RqDecomposition::empty();
    let mut rw = LinalgWorkspace::new();
    RqDecomposition::factor_into(
        a.view(),
        &scioxide_linalg::LinalgExecutionOptions::new().with_execution_policy(policy),
        &mut rq,
        &mut rw,
        context,
    )
    .unwrap();
    close(&multiply(rq.r_matrix(), rq.q_matrix()), &a, 1);
}

fn scale_and_basis<S: NumericScalar>() {
    let a = matrix::<S>(6, 3, 4);
    let opts = options::<S>();
    let range = if S::Real::EPSILON.into_f64() > 1e-10 {
        30
    } else {
        300
    };
    for exponent in [-range, range] {
        let scale = 2.0_f64.powi(exponent);
        let scaled = Matrix::from_vec(
            6,
            3,
            a.as_slice().iter().map(|&x| x * real::<S>(scale)).collect(),
        )
        .unwrap();
        let rq = RqDecomposition::factor(scaled.view()).unwrap();
        let reconstructed = multiply(rq.r_matrix(), rq.q_matrix());
        let normalized = Matrix::from_vec(
            6,
            3,
            reconstructed
                .as_slice()
                .iter()
                .map(|&x| x / real::<S>(scale))
                .collect(),
        )
        .unwrap();
        close(&normalized, &a, 1);
        let mut f = QrUpdateDecomposition::factor_with_options(scaled.view(), &(opts)).unwrap();
        let mut w = LinalgWorkspace::new();
        let u = Matrix::from_vec(
            6,
            1,
            (0..6)
                .map(|i| real::<S>((i + 1) as f64 * scale / 16.0))
                .collect(),
        )
        .unwrap();
        let v = matrix::<S>(3, 1, 2);
        let edit = QrModification::RankUpdate {
            u: u.view(),
            v: v.view(),
        };
        f.modify_in_place(
            edit,
            &(opts),
            &mut w,
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap();
        let target = edited(&scaled, edit);
        verify(&f, &target, 1);
        let product = multiply(f.q_matrix(), f.r_matrix());
        let normalized = Matrix::from_vec(
            6,
            3,
            product
                .as_slice()
                .iter()
                .map(|&x| x / real::<S>(scale))
                .collect(),
        )
        .unwrap();
        let expected = Matrix::from_vec(
            6,
            3,
            (0..18)
                .map(|ij| {
                    target.view().get(ij / 3, f.permutation()[ij % 3]).unwrap() / real::<S>(scale)
                })
                .collect(),
        )
        .unwrap();
        close(&normalized, &expected, 1);
    }
    // Unitary changes of the paired update basis leave UV^H invariant.
    let t = Matrix::from_vec(
        2,
        2,
        vec![
            real::<S>(0.6),
            real::<S>(-0.8),
            real::<S>(0.8),
            real::<S>(0.6),
        ],
    )
    .unwrap();
    let u = matrix::<S>(6, 2, 3);
    let v = matrix::<S>(3, 2, 7);
    let ut = multiply(&u, &t);
    let vt = multiply(&v, &t);
    let mut f = QrUpdateDecomposition::factor_with_options(a.view(), &(opts)).unwrap();
    let mut g = f.clone();
    let mut w = LinalgWorkspace::new();
    f.modify_in_place(
        QrModification::RankUpdate {
            u: u.view(),
            v: v.view(),
        },
        &(opts),
        &mut w,
        scioxide_core::ExecutionContext::sequential(),
    )
    .unwrap();
    g.modify_in_place(
        QrModification::RankUpdate {
            u: ut.view(),
            v: vt.view(),
        },
        &(opts),
        &mut w,
        scioxide_core::ExecutionContext::sequential(),
    )
    .unwrap();
    let projector = |f: &QrUpdateDecomposition<S>| {
        let q = Matrix::from_vec(
            6,
            3,
            (0..18)
                .map(|ij| f.q_matrix().view().get(ij / 3, ij % 3).unwrap())
                .collect(),
        )
        .unwrap();
        multiply(&q, &q.view().adjoint().try_to_matrix().unwrap())
    };
    close(&projector(&f), &projector(&g), 2);
    assert_eq!(
        f.rank_decision().report().rank(),
        g.rank_decision().report().rank()
    );
    // RQ row-space projectors are invariant under a left unitary basis change.
    let b = a.view().transpose().try_to_matrix().unwrap();
    let t = Matrix::from_vec(
        3,
        3,
        vec![
            real::<S>(0.6),
            real::<S>(-0.8),
            S::ZERO,
            real::<S>(0.8),
            real::<S>(0.6),
            S::ZERO,
            S::ZERO,
            S::ZERO,
            S::ONE,
        ],
    )
    .unwrap();
    let tb = multiply(&t, &b);
    let row_projector = |b: &Matrix<S>| {
        let f = RqDecomposition::factor(b.view()).unwrap();
        let q = Matrix::from_vec(
            3,
            6,
            (0..18)
                .map(|ij| f.q_matrix().view().get(ij / 6 + 3, ij % 6).unwrap())
                .collect(),
        )
        .unwrap();
        multiply(&q.view().adjoint().try_to_matrix().unwrap(), &q)
    };
    close(&row_projector(&b), &row_projector(&tb), 1);
}
#[test]
fn scales_and_invariant_subspaces() {
    scale_and_basis::<f32>();
    scale_and_basis::<f64>();
    scale_and_basis::<Complex<f32>>();
    scale_and_basis::<Complex<f64>>();
}

#[test]
fn rq_failures_views_empty_deletions_and_rank_policy() {
    use scioxide_core::ExecutionPolicy;
    use scioxide_linalg::{LinalgWorkspace, MatrixView};
    let a = matrix::<Complex<f64>>(5, 4, 3);
    let mut backing = vec![Complex::new(0.0, 0.0); 80];
    for i in 0..5 {
        for j in 0..4 {
            backing[i * 16 + j * 2] = a.view().get(i, j).unwrap().conj();
        }
    }
    let view = MatrixView::from_slice_with_strides(5, 4, &backing, 16, 2)
        .unwrap()
        .conjugate();
    let mut rq = RqDecomposition::factor(view).unwrap();
    close(&multiply(rq.r_matrix(), rq.q_matrix()), &a, 1);
    let before = rq.clone();
    let mut rw = LinalgWorkspace::new();
    let bad = Matrix::from_vec(1, 1, vec![Complex::new(f64::INFINITY, 0.0)]).unwrap();
    assert!(matches!(
        (RqDecomposition::factor_into(
            bad.view(),
            &scioxide_linalg::LinalgExecutionOptions::new()
                .with_execution_policy(ExecutionPolicy::Sequential),
            &mut rq,
            &mut rw,
            scioxide_core::ExecutionContext::sequential()
        ))
        .as_ref()
        .err()
        .unwrap()
        .view(),
        scioxide_core::SciOxideErrorView::NonFinite { .. }
    ));
    assert_eq!(rq.q_matrix(), before.q_matrix());
    assert_eq!(rq.r_matrix(), before.r_matrix());
    let mut f =
        QrUpdateDecomposition::factor_with_options(view, &(options::<Complex<f64>>())).unwrap();
    verify(&f, &a, 1);
    let mut w = LinalgWorkspace::new();
    f.modify_in_place(
        QrModification::DeleteRows { index: 0, count: 5 },
        &(options::<Complex<f64>>()),
        &mut w,
        scioxide_core::ExecutionContext::sequential(),
    )
    .unwrap();
    verify(&f, &Matrix::zeros(0, 4).unwrap(), 1);
    f.modify_in_place(
        QrModification::DeleteColumns { index: 0, count: 4 },
        &(options::<Complex<f64>>()),
        &mut w,
        scioxide_core::ExecutionContext::sequential(),
    )
    .unwrap();
    verify(&f, &Matrix::zeros(0, 0).unwrap(), 1);
    let a = Matrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, 1e-18]).unwrap();
    let mut f = QrUpdateDecomposition::factor_with_options(
        a.view(),
        &(QrUpdateOptions::new(RankPolicy::Exact)),
    )
    .unwrap();
    assert_eq!(f.rank_decision().report().rank(), 2);
    let report = f
        .modify_in_place(
            QrModification::DeleteRows { index: 2, count: 0 },
            &(options::<f64>()),
            &mut LinalgWorkspace::new(),
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(report.before().report().rank(), 1);
    assert_eq!(report.after().report().rank(), 1);
    verify(&f, &a, 1);
}

fn rq_rank_deficient<S: NumericScalar>() {
    use scioxide_core::ExecutionPolicy;
    use scioxide_linalg::LinalgWorkspace;
    let mut out = RqDecomposition::empty();
    let mut work = LinalgWorkspace::new();
    for (m, n) in [(4, 4), (7, 3), (3, 7)] {
        for zero in [false, true] {
            let mut a = matrix::<S>(m, n, 4);
            if zero {
                a.as_mut_slice().fill(S::ZERO);
            } else {
                // An exact dependent column, independent of pivot selection.
                for i in 0..m {
                    let value = a.view().get(i, 0).unwrap();
                    a.view_mut().set(i, n - 1, value).unwrap();
                }
            }
            RqDecomposition::factor_into(
                a.view(),
                &scioxide_linalg::LinalgExecutionOptions::new()
                    .with_execution_policy(ExecutionPolicy::Sequential),
                &mut out,
                &mut work,
                scioxide_core::ExecutionContext::sequential(),
            )
            .unwrap();
            close(&multiply(out.r_matrix(), out.q_matrix()), &a, 1);
            unitary(out.q_matrix(), 1);
            for i in 0..m {
                for j in 0..n {
                    if i + n > j + m {
                        assert_eq!(out.r_matrix().view().get(i, j).unwrap(), S::ZERO);
                    }
                }
            }
        }
    }
}
#[test]
fn rq_repeated_rank_deficient_and_zero_factors() {
    rq_rank_deficient::<f32>();
    rq_rank_deficient::<f64>();
    rq_rank_deficient::<Complex<f32>>();
    rq_rank_deficient::<Complex<f64>>();
}

#[test]
fn long_rotation_sequences() {
    let mut a = matrix::<Complex<f64>>(69, 35, 7);
    let mut factor =
        QrUpdateDecomposition::factor_with_options(a.view(), &(options::<Complex<f64>>())).unwrap();
    let mut work = LinalgWorkspace::new();
    let u = matrix::<Complex<f64>>(69, 2, 9);
    let v = matrix::<Complex<f64>>(35, 2, 11);
    for edit in [
        QrModification::RankUpdate {
            u: u.view(),
            v: v.view(),
        },
        QrModification::RankDowndate {
            u: u.view(),
            v: v.view(),
        },
        QrModification::DeleteRows {
            index: 17,
            count: 2,
        },
    ] {
        a = edited(&a, edit);
        factor
            .modify_in_place(
                edit,
                &(options::<Complex<f64>>()),
                &mut work,
                scioxide_core::ExecutionContext::sequential(),
            )
            .unwrap();
        verify(&factor, &a, 3);
    }
}

fn rq_panel_boundaries<S: NumericScalar>() {
    for n in [33, 65, 97] {
        for (m, n) in [(n + 3, n), (n, n + 3)] {
            let a = matrix::<S>(m, n, 17);
            let factor = RqDecomposition::factor(a.view()).unwrap();
            close(&multiply(factor.r_matrix(), factor.q_matrix()), &a, 1);
            unitary(factor.q_matrix(), 1);
            for i in 0..m {
                for j in 0..n {
                    if i + n > j + m {
                        assert_eq!(factor.r_matrix().view().get(i, j).unwrap(), S::ZERO);
                    }
                }
            }
        }
    }
}

#[test]
fn rq_backward_panels_cover_rectangular_scalar_domains() {
    rq_panel_boundaries::<f32>();
    rq_panel_boundaries::<f64>();
    rq_panel_boundaries::<Complex<f32>>();
    rq_panel_boundaries::<Complex<f64>>();
}

#[test]
fn rq_late_panel_overflow_preserves_publication_and_allows_retry() {
    use scioxide_core::ExecutionPolicy;
    use scioxide_linalg::LinalgWorkspace;
    let a = Matrix::<f64>::identity(73).unwrap();
    let mut factor = RqDecomposition::factor(a.view()).unwrap();
    let before = factor.clone();
    let mut work = LinalgWorkspace::new();
    let mut overflow = a.clone();
    overflow.view_mut().set(0, 0, f64::MAX).unwrap();
    overflow.view_mut().set(1, 0, f64::MAX).unwrap();
    overflow.view_mut().set(1, 1, f64::MAX).unwrap();
    assert!(
        RqDecomposition::factor_into(
            overflow.view(),
            &scioxide_linalg::LinalgExecutionOptions::new()
                .with_execution_policy(ExecutionPolicy::Sequential),
            &mut factor,
            &mut work,
            scioxide_core::ExecutionContext::sequential()
        )
        .is_err()
    );
    assert_eq!(factor.r_matrix(), before.r_matrix());
    assert_eq!(factor.q_matrix(), before.q_matrix());
    RqDecomposition::factor_into(
        a.view(),
        &scioxide_linalg::LinalgExecutionOptions::new()
            .with_execution_policy(ExecutionPolicy::Sequential),
        &mut factor,
        &mut work,
        scioxide_core::ExecutionContext::sequential(),
    )
    .unwrap();
    close(&multiply(factor.r_matrix(), factor.q_matrix()), &a, 1);
}

// Full-rank pseudo-random entries in [-1, 1) from a 64-bit LCG.
fn random<S: NumericScalar>(m: usize, n: usize, seed: u64) -> Matrix<S> {
    let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    let mut next = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    };
    Matrix::from_vec(
        m,
        n,
        (0..m * n)
            .map(|_| {
                let re = next();
                let im = if S::IS_REAL { 0.0 } else { next() };
                S::from_components(
                    S::Real::try_from_f64_lossy(re).unwrap(),
                    S::Real::try_from_f64_lossy(im).unwrap(),
                )
            })
            .collect(),
    )
    .unwrap()
}
fn threshold<S: NumericScalar>(relative: f64) -> RankPolicy<S::Real> {
    RankPolicy::Threshold(
        scioxide_core::RankThreshold::new(
            S::Real::try_from_f64_lossy(0.0).unwrap(),
            S::Real::try_from_f64_lossy(relative).unwrap(),
        )
        .unwrap(),
    )
}
// Rank of an explicitly formed matrix from its singular values.
fn reference_rank<S: NumericScalar>(a: &Matrix<S>, policy: RankPolicy<S::Real>) -> usize {
    scioxide_linalg::SvdValues::compute(a.view())
        .unwrap()
        .analyze_rank(policy)
        .unwrap()
        .rank()
}
fn modify<S: NumericScalar>(
    f: &mut QrUpdateDecomposition<S>,
    edit: QrModification<'_, S>,
    policy: RankPolicy<S::Real>,
) -> scioxide_linalg::QrUpdateReport<S::Real> {
    f.modify_in_place(
        edit,
        &QrUpdateOptions::new(policy),
        &mut LinalgWorkspace::new(),
        scioxide_core::ExecutionContext::sequential(),
    )
    .unwrap()
}

fn downdate_rank_drop<S: NumericScalar>() {
    use scioxide_linalg::QrRankSource::Estimator;
    let policy = threshold::<S>(1e-9);
    for (m, n) in [(6, 6), (9, 5)] {
        let a = random::<S>(m, n, 3);
        assert_eq!(reference_rank(&a, policy), n);
        // `A - (A x) x^H / |x|^2` annihilates x.
        let x = random::<S>(n, 1, 5);
        let u = multiply(&a, &x);
        let square: f64 = x
            .as_slice()
            .iter()
            .map(|x| x.abs().into_f64().powi(2))
            .sum();
        let v = Matrix::from_vec(
            n,
            1,
            x.as_slice()
                .iter()
                .map(|&x| x * real::<S>(1.0 / square))
                .collect(),
        )
        .unwrap();
        let mut f =
            QrUpdateDecomposition::factor_with_options(a.view(), &QrUpdateOptions::new(policy))
                .unwrap();
        assert_eq!(f.rank_decision().report().rank(), n);
        let edit = QrModification::RankDowndate {
            u: u.view(),
            v: v.view(),
        };
        let target = edited(&a, edit);
        let dropped = modify(&mut f, edit, policy);
        verify(&f, &target, 1);
        assert_eq!(reference_rank(&target, policy), n - 1);
        assert!(dropped.rank_lost());
        assert_eq!(dropped.before().report().rank(), n);
        assert_eq!(dropped.after().report().rank(), n - 1);
        assert_eq!(dropped.after().report().maximum_rank(), n);
        assert_eq!(dropped.after().source(), Estimator);
        let restored = modify(
            &mut f,
            QrModification::RankUpdate {
                u: u.view(),
                v: v.view(),
            },
            policy,
        );
        verify(&f, &a, 2);
        // The previous decision is reused under the same policy.
        assert_eq!(restored.before(), dropped.after());
        assert_eq!(restored.after().report().rank(), n);
        assert_eq!(restored.after().source(), Estimator);
    }
}
#[test]
fn downdates_that_drop_rank_are_estimated() {
    downdate_rank_drop::<f64>();
    downdate_rank_drop::<Complex<f64>>();
}

// Column j of `a` scaled by 2^(-step j).
fn graded<S: NumericScalar>(a: &Matrix<S>, step: i32, offset: i32) -> Matrix<S> {
    Matrix::from_vec(
        a.rows(),
        a.cols(),
        a.as_slice()
            .iter()
            .enumerate()
            .map(|(ij, &x)| x * real::<S>(2.0_f64.powi(-step * (ij % a.cols()) as i32 - offset)))
            .collect(),
    )
    .unwrap()
}
fn graded_columns<S: NumericScalar>() {
    let policy = RankPolicy::DimensionScaledEpsilon;
    let mut a = graded(&random::<S>(8, 5, 6), 8, 0);
    let mut f = QrUpdateDecomposition::factor_with_options(a.view(), &QrUpdateOptions::new(policy))
        .unwrap();
    assert_eq!(f.rank_decision().report().rank(), 5);
    let middle = graded(&random::<S>(8, 2, 12), 4, 20);
    let tiny = graded(&random::<S>(8, 1, 11), 0, 60);
    let last = graded(&random::<S>(8, 1, 13), 0, 40);
    for (step, edit) in [
        QrModification::InsertColumns {
            index: 2,
            values: middle.view(),
        },
        QrModification::DeleteColumns { index: 3, count: 2 },
        QrModification::InsertColumns {
            index: 0,
            values: tiny.view(),
        },
        QrModification::DeleteColumns { index: 0, count: 1 },
        QrModification::DeleteColumns { index: 4, count: 1 },
        QrModification::InsertColumns {
            index: 4,
            values: last.view(),
        },
    ]
    .into_iter()
    .enumerate()
    {
        a = edited(&a, edit);
        let report = modify(&mut f, edit, policy);
        verify(&f, &a, step + 1);
        let rank = report.after().report();
        assert_eq!(rank.rank(), reference_rank(&a, policy), "step {step}");
        assert_eq!(rank.maximum_rank(), a.cols(), "step {step}");
        assert_eq!(
            report.after().source(),
            scioxide_linalg::QrRankSource::Estimator,
            "step {step}"
        );
    }
    // Only the inserted column below the threshold was not counted.
    assert_eq!(f.rank_decision().report().rank(), 5);
}
#[test]
fn graded_column_insertions_and_deletions_track_rank() {
    graded_columns::<f64>();
    graded_columns::<Complex<f64>>();
}

#[test]
fn householder_conversion_reserves_tall_q_formation() {
    // Forming the explicit m x m Q of a tall factor needs more scratch than
    // the update's own factorization.
    for (m, n) in [(3, 1), (10, 2), (12, 5)] {
        let a = matrix::<f64>(m, n, 3);
        let packed = scioxide_linalg::HouseholderDecomposition::factor(
            &a,
            scioxide_linalg::HouseholderOrientation::Qr,
            scioxide_linalg::HouseholderPivoting::Column,
        )
        .unwrap();
        let f = QrUpdateDecomposition::from_householder_with_options(&packed, &options::<f64>())
            .unwrap();
        verify(&f, &a, 1);
    }
}
