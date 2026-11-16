//! Independent residual and direct-solution tests for biorthogonal methods.
#![allow(clippy::unwrap_used)]
use scioxide_core::AlgebraicScalar;
use scioxide_core::{Budget, Complex, NumericScalar, RealScalar, ResidualTolerance};
use scioxide_linalg::Preconditioning;
use scioxide_linalg::{
    BicgSolver, CgsSolver, LinearSolveTermination, LuDecomposition, Matrix, QmrSolver, TfqmrSolver,
    Vector,
};
fn scalar<S: NumericScalar>(r: f64, i: f64) -> S {
    S::from_components(
        S::Real::try_from_f64_lossy(r).unwrap(),
        S::Real::try_from_f64_lossy(i).unwrap(),
    )
}
fn qualify<S: NumericScalar>() {
    let a = Matrix::from_vec(
        4,
        4,
        vec![
            scalar::<S>(4., 0.2),
            scalar(2., -0.3),
            scalar(0., 0.),
            scalar(1., 0.1),
            scalar(-1., 0.),
            scalar(5., -0.4),
            scalar(2., 0.3),
            scalar(0., 0.),
            scalar(0., 0.),
            scalar(-2., 0.2),
            scalar(-3., 0.1),
            scalar(1., 0.),
            scalar(0., 0.2),
            scalar(0., 0.),
            scalar(1., 0.),
            scalar(4., 0.2),
        ],
    )
    .unwrap();
    let exact = Vector::from_vec(vec![
        scalar(1., 0.2),
        scalar(-2., 0.3),
        scalar(0.5, -0.4),
        scalar(3., 0.1),
    ]);
    let b = a.view().mul_vec(&exact).unwrap();
    let direct = LuDecomposition::factor(&a).unwrap().solve(&b).unwrap();
    let eps = S::Real::EPSILON;
    let tol = ResidualTolerance::new(
        eps * S::Real::try_from_f64_lossy(100.).unwrap(),
        S::Real::ZERO,
    )
    .unwrap();
    let limit = Budget::try_from(200).unwrap();
    let reports = [
        BicgSolver::new(tol, limit)
            .solve_with_options(
                &a.view().try_finite().unwrap(),
                &b,
                Preconditioning::None,
                None,
            )
            .unwrap(),
        CgsSolver::new(tol, limit)
            .solve_with_options(
                &a.view().try_finite().unwrap(),
                &b,
                Preconditioning::None,
                None,
            )
            .unwrap(),
        QmrSolver::new(tol, limit)
            .solve_with_options(
                &a.view().try_finite().unwrap(),
                &b,
                Preconditioning::None,
                None,
            )
            .unwrap(),
        TfqmrSolver::new(tol, limit)
            .solve_with_options(
                &a.view().try_finite().unwrap(),
                &b,
                Preconditioning::None,
                None,
            )
            .unwrap(),
    ];
    for (method, report) in reports.iter().enumerate() {
        assert_eq!(
            report.report().termination(),
            LinearSolveTermination::Converged,
            "method {method}: {report:?}"
        );
        let mut rn = S::Real::ZERO;
        for row in 0..4 {
            let mut sum = S::ZERO;
            for col in 0..4 {
                sum = sum + a.view().get(row, col).unwrap() * report.solution().as_slice()[col];
            }
            rn = rn.hypot((b.as_slice()[row] - sum).abs());
        }
        assert!(
            tol.is_satisfied(
                rn,
                b.view()
                    .norm2_scoped(
                        &scioxide_linalg::LinalgExecutionOptions::new(),
                        scioxide_core::ExecutionContext::sequential()
                    )
                    .unwrap()
            )
            .unwrap()
        );
        let error = report
            .solution()
            .view()
            .try_sub(&direct)
            .unwrap()
            .view()
            .norm2_scoped(
                &scioxide_linalg::LinalgExecutionOptions::new(),
                scioxide_core::ExecutionContext::sequential(),
            )
            .unwrap();
        assert!(error <= eps * S::Real::try_from_f64_lossy(2000.).unwrap());
    }
}
#[test]
fn real32() {
    qualify::<f32>();
}
#[test]
fn real64() {
    qualify::<f64>();
}
#[test]
fn complex32() {
    qualify::<Complex<f32>>();
}
#[test]
fn complex64() {
    qualify::<Complex<f64>>();
}

use scioxide_linalg::{
    GaussSeidelSolver, JacobiSolver, RelaxationWeight, RichardsonSolver, SorSolver, SorWeight,
    StationaryMatrix,
};
fn stationary<S: NumericScalar>() {
    let a = Matrix::from_vec(
        3,
        3,
        vec![
            scalar::<S>(5., 0.),
            scalar(-1., 0.25),
            scalar(0., 0.),
            scalar(-1., -0.25),
            scalar(4., 0.),
            scalar(-1., 0.25),
            scalar(0., 0.),
            scalar(-1., -0.25),
            scalar(5., 0.),
        ],
    )
    .unwrap();
    let exact = Vector::from_vec(vec![scalar(1., 0.3), scalar(2., -0.1), scalar(-1., 0.2)]);
    let b = a.view().mul_vec(&exact).unwrap();
    let tol = ResidualTolerance::new(
        S::Real::EPSILON * S::Real::try_from_f64_lossy(50.).unwrap(),
        S::Real::ZERO,
    )
    .unwrap();
    let limit = Budget::try_from(500).unwrap();
    let csr = scioxide_sparse::CsrMatrix::from_raw(
        3,
        3,
        vec![0, 3, 6, 9],
        vec![0, 1, 2, 0, 1, 2, 0, 1, 2],
        a.as_slice().to_vec(),
    )
    .unwrap();
    let csc = csr.to_csc().unwrap();
    for input in [
        StationaryMatrix::Dense(a.view().try_finite().unwrap()),
        StationaryMatrix::Csr(&csr),
        StationaryMatrix::Csc(&csc),
    ] {
        let reports = [
            JacobiSolver::new(tol, limit)
                .with_weight(
                    RelaxationWeight::new(S::Real::try_from_f64_lossy(0.8).unwrap()).unwrap(),
                )
                .solve_with_options(input, &b, None)
                .unwrap(),
            GaussSeidelSolver::new(tol, limit)
                .solve_with_options(input, &b, None)
                .unwrap(),
            SorSolver::new(
                tol,
                limit,
                SorWeight::new(S::Real::try_from_f64_lossy(1.2).unwrap()).unwrap(),
            )
            .solve_with_options(input, &b, None)
            .unwrap(),
        ];
        for report in reports {
            assert_eq!(
                report.report().termination(),
                LinearSolveTermination::Converged
            );
            assert!(
                tol.is_satisfied(
                    b.view()
                        .try_sub(&a.view().mul_vec(report.solution()).unwrap())
                        .unwrap()
                        .view()
                        .norm2_scoped(
                            &scioxide_linalg::LinalgExecutionOptions::new(),
                            scioxide_core::ExecutionContext::sequential()
                        )
                        .unwrap(),
                    b.view()
                        .norm2_scoped(
                            &scioxide_linalg::LinalgExecutionOptions::new(),
                            scioxide_core::ExecutionContext::sequential()
                        )
                        .unwrap()
                )
                .unwrap()
            );
        }
    }
    let report = RichardsonSolver::new(
        tol,
        limit,
        RelaxationWeight::new(S::Real::try_from_f64_lossy(0.15).unwrap()).unwrap(),
    )
    .solve_with_options(
        &a.view().try_finite().unwrap(),
        &b,
        Preconditioning::None,
        None,
    )
    .unwrap();
    assert!(report.is_success());
}
#[test]
fn stationary_real32() {
    stationary::<f32>();
}
#[test]
fn stationary_real64() {
    stationary::<f64>();
}
#[test]
fn stationary_complex32() {
    stationary::<Complex<f32>>();
}
#[test]
fn stationary_complex64() {
    stationary::<Complex<f64>>();
}

#[test]
fn stationary_domains_and_nonconvergence() {
    let tol = ResidualTolerance::new(1e-12, 0.).unwrap();
    let limit = Budget::try_from(10).unwrap();
    for omega in [0., -1., f64::NAN, f64::INFINITY] {
        assert!(RelaxationWeight::new(omega).is_err());
        assert!(SorWeight::new(omega).is_err());
    }
    assert!(SorWeight::new(2.).is_err());
    assert!(RelaxationWeight::new(2.).is_ok());
    let a = Matrix::from_vec(2, 2, vec![1., 2., 2., 1.]).unwrap();
    let b = Vector::from_vec(vec![1., 1.]);
    for report in [
        JacobiSolver::new(tol, limit)
            .solve_with_options(
                StationaryMatrix::Dense(a.view().try_finite().unwrap()),
                &b,
                None,
            )
            .unwrap(),
        GaussSeidelSolver::new(tol, limit)
            .solve_with_options(
                StationaryMatrix::Dense(a.view().try_finite().unwrap()),
                &b,
                None,
            )
            .unwrap(),
        RichardsonSolver::new(tol, limit, RelaxationWeight::one())
            .solve_with_options(
                &a.view().try_finite().unwrap(),
                &b,
                Preconditioning::None,
                None,
            )
            .unwrap(),
    ] {
        assert_eq!(
            report.report().termination(),
            LinearSolveTermination::IterationLimit
        );
        assert!(!report.is_success());
    }
    let a = Matrix::from_vec(2, 2, vec![0., 1., 1., 1.]).unwrap();
    assert!(matches!(
        (JacobiSolver::new(tol, limit).solve_with_options(
            StationaryMatrix::Dense(a.view().try_finite().unwrap()),
            &b,
            None,
        ))
        .as_ref()
        .err()
        .unwrap()
        .view(),
        scioxide_core::SciOxideErrorView::Singular { .. }
    ));
}

#[test]
fn scaled_and_long_systems() {
    for scale in [1e-100, 1., 1e100] {
        let n = 64;
        let mut data = vec![0.; n * n];
        for i in 0..n {
            data[i * n + i] = (1. + i as f64 / 3.) * scale;
            if i + 1 < n {
                data[i * n + i + 1] = 0.25 * scale;
            }
        }
        let a = Matrix::from_vec(n, n, data).unwrap();
        let exact = Vector::from_vec(vec![1.; n]);
        let b = a.view().mul_vec(&exact).unwrap();
        let tol = ResidualTolerance::new(0., 1e-11).unwrap();
        let limit = Budget::try_from(500).unwrap();
        for result in [
            BicgSolver::new(tol, limit)
                .solve_with_options(
                    &a.view().try_finite().unwrap(),
                    &b,
                    Preconditioning::None,
                    None,
                )
                .unwrap(),
            CgsSolver::new(tol, limit)
                .solve_with_options(
                    &a.view().try_finite().unwrap(),
                    &b,
                    Preconditioning::None,
                    None,
                )
                .unwrap(),
            QmrSolver::new(tol, limit)
                .solve_with_options(
                    &a.view().try_finite().unwrap(),
                    &b,
                    Preconditioning::None,
                    None,
                )
                .unwrap(),
            TfqmrSolver::new(tol, limit)
                .solve_with_options(
                    &a.view().try_finite().unwrap(),
                    &b,
                    Preconditioning::None,
                    None,
                )
                .unwrap(),
        ] {
            assert!(result.is_success(), "{scale}: {result:?}");
        }
    }
}

#[test]
fn nonsingular_breakdown_is_distinct_from_accuracy() {
    // Invertible skew-symmetric A has r^T A r=0: an initial BiCG-family breakdown.
    let a = Matrix::from_vec(2, 2, vec![0., 1., -1., 0.]).unwrap();
    let b = Vector::from_vec(vec![1., 0.]);
    let direct = LuDecomposition::factor(&a).unwrap().solve(&b).unwrap();
    assert_eq!(direct.as_slice(), &[0., 1.]);
    let tol = ResidualTolerance::new(1e-12, 0.).unwrap();
    let budget = Budget::try_from(8).unwrap();
    for result in [
        BicgSolver::new(tol, budget)
            .solve_with_options(
                &a.view().try_finite().unwrap(),
                &b,
                Preconditioning::None,
                None,
            )
            .unwrap(),
        CgsSolver::new(tol, budget)
            .solve_with_options(
                &a.view().try_finite().unwrap(),
                &b,
                Preconditioning::None,
                None,
            )
            .unwrap(),
        QmrSolver::new(tol, budget)
            .solve_with_options(
                &a.view().try_finite().unwrap(),
                &b,
                Preconditioning::None,
                None,
            )
            .unwrap(),
        TfqmrSolver::new(tol, budget)
            .solve_with_options(
                &a.view().try_finite().unwrap(),
                &b,
                Preconditioning::None,
                None,
            )
            .unwrap(),
    ] {
        assert!(matches!(
            result.report().termination(),
            LinearSolveTermination::Breakdown(_)
        ));
        assert!(!result.is_success());
        assert_eq!(result.report().true_residual().unwrap().raw_norm(), 1.);
    }
}

#[test]
fn ill_conditioned_diagonal_true_residual_and_backward_error() {
    let a = Matrix::from_vec(3, 3, vec![1., 0., 0., 0., 1e-4, 0., 0., 0., 1e-8]).unwrap();
    let b = Vector::from_vec(vec![1., 1e-4, 1e-8]);
    let tol = ResidualTolerance::new(1e-13, 0.).unwrap();
    let budget = Budget::try_from(200).unwrap();
    for result in [
        BicgSolver::new(tol, budget)
            .solve_with_options(
                &a.view().try_finite().unwrap(),
                &b,
                Preconditioning::None,
                None,
            )
            .unwrap(),
        CgsSolver::new(tol, budget)
            .solve_with_options(
                &a.view().try_finite().unwrap(),
                &b,
                Preconditioning::None,
                None,
            )
            .unwrap(),
        QmrSolver::new(tol, budget)
            .solve_with_options(
                &a.view().try_finite().unwrap(),
                &b,
                Preconditioning::None,
                None,
            )
            .unwrap(),
        TfqmrSolver::new(tol, budget)
            .solve_with_options(
                &a.view().try_finite().unwrap(),
                &b,
                Preconditioning::None,
                None,
            )
            .unwrap(),
    ] {
        assert!(result.is_success(), "{result:?}");
        let residual = result
            .solution()
            .as_slice()
            .iter()
            .zip([1., 1e-4, 1e-8])
            .fold(0_f64, |m, (&x, d)| m.max((d - d * x).abs()));
        let xn = result
            .solution()
            .as_slice()
            .iter()
            .fold(0_f64, |m, x| m.max(x.abs()));
        let backward_error = residual / (xn + 1.);
        assert!(backward_error < 1e-13);
    }
}
