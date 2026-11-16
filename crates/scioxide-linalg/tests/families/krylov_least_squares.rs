//! Independent exact fixtures for the short-recurrence Krylov methods.
#![allow(clippy::unwrap_used)]
use scioxide_core::AdjointOperator;
use scioxide_core::AlgebraicScalar;
use scioxide_core::{
    Budget, Complex, GradientTolerance, NumericScalar, RealScalar, ResidualTolerance,
};
use scioxide_linalg::Preconditioning;
use scioxide_linalg::{
    IterativeSolveResult, LinearSolveTermination, LsmrSolver, LsqrSolver, Matrix, MinresSolver,
    Vector,
};

fn budget() -> Budget {
    Budget::try_from(300).unwrap()
}
fn real<S: NumericScalar>(value: f64) -> S {
    S::from_real(S::Real::try_from_f64_lossy(value).unwrap())
}
fn tolerance<S: NumericScalar>() -> ResidualTolerance<S::Real> {
    ResidualTolerance::new(
        S::Real::ZERO,
        S::Real::SQRT_EPSILON * S::Real::try_from_f64_lossy(0.1).unwrap(),
    )
    .unwrap()
}
fn normal<S: NumericScalar>() -> GradientTolerance<S::Real> {
    GradientTolerance::from_tolerance(tolerance::<S>().tolerance())
}
fn least_squares<S: NumericScalar>(
    a: &dyn AdjointOperator<S>,
    b: &Vector<S>,
) -> [IterativeSolveResult<S>; 2] {
    [
        LsqrSolver::new(tolerance::<S>(), budget())
            .with_normal_tolerance(normal::<S>())
            .solve_with_options(a, b, None)
            .unwrap(),
        LsmrSolver::new(tolerance::<S>(), budget())
            .with_normal_tolerance(normal::<S>())
            .solve_with_options(a, b, None)
            .unwrap(),
    ]
}
fn check<S: NumericScalar>(a: &Matrix<S>, b: &Vector<S>, result: &IterativeSolveResult<S>) {
    assert!(
        result.is_success(),
        "{:?}; shape={}x{} rhs={:?}",
        result.report(),
        a.rows(),
        a.cols(),
        b
    );
    // Independent loops, separate from the production matvec and reduction kernels.
    let mut residual = vec![S::ZERO; a.rows()];
    for (row, r) in residual.iter_mut().enumerate() {
        *r = b.as_slice()[row];
        for (col, &x) in result.solution().as_slice().iter().enumerate() {
            *r = *r - a.view().get(row, col).unwrap() * x;
        }
    }
    let mut gradient = vec![S::ZERO; a.cols()];
    for (col, g) in gradient.iter_mut().enumerate() {
        for (row, &r) in residual.iter().enumerate() {
            *g = *g + a.view().get(row, col).unwrap().conj() * r;
        }
    }
    let direct_norm = |values: &[S]| {
        values
            .iter()
            .map(|v| v.abs().try_to_f64_exact().unwrap().powi(2))
            .sum::<f64>()
            .sqrt()
    };
    let r = direct_norm(&residual);
    let g = direct_norm(&gradient);
    let epsilon = S::Real::EPSILON.try_to_f64_exact().unwrap().sqrt();
    assert!(
        (result
            .report()
            .true_residual()
            .unwrap()
            .raw_norm()
            .try_to_f64_exact()
            .unwrap()
            - r)
            .abs()
            < epsilon * (1.0 + r)
    );
    assert!(
        (result
            .report()
            .krylov()
            .unwrap()
            .normal_residual()
            .unwrap()
            .raw_norm()
            .try_to_f64_exact()
            .unwrap()
            - g)
            .abs()
            < epsilon * (1.0 + g)
    );
    assert!(g < 100.0 * epsilon, "normal residual {g}");
}
fn exact_fixtures<S: NumericScalar>(phase: S) {
    // A has full column rank; the final coordinate supplies an exactly orthogonal residual.
    let a = Matrix::from_vec(
        4,
        2,
        vec![
            phase,
            real::<S>(0.),
            real::<S>(0.),
            real::<S>(2.),
            phase,
            real::<S>(2.),
            real::<S>(0.),
            real::<S>(0.),
        ],
    )
    .unwrap();
    for noise in [0.0, 3.0] {
        let b = Vector::from_vec(vec![
            phase * real::<S>(2.),
            real::<S>(-2.),
            phase * real::<S>(2.) - real::<S>(2.),
            real::<S>(noise),
        ]);
        let csr = a
            .to_sparse_csr(scioxide_sparse::ExplicitZeroPolicy::Drop)
            .unwrap();
        let csc = csr.to_csc().unwrap();
        let view = a.view();
        let operator = IndependentOperator(&a);
        for result in [
            &a.view().try_finite().unwrap() as &dyn AdjointOperator<S>,
            &view.try_finite().unwrap(),
            &csr,
            &csc,
            &operator,
        ]
        .into_iter()
        .flat_map(|op| least_squares(op, &b))
        {
            check(&a, &b, &result);
            assert!(
                (result.solution().as_slice()[0] - real::<S>(2.)).abs()
                    < S::Real::SQRT_EPSILON * S::Real::try_from_f64_lossy(10.).unwrap()
            );
            assert!(
                (result.solution().as_slice()[1] - real::<S>(-1.)).abs()
                    < S::Real::SQRT_EPSILON * S::Real::try_from_f64_lossy(10.).unwrap()
            );
            if noise != 0. {
                assert_eq!(
                    result.report().termination(),
                    LinearSolveTermination::LeastSquaresConverged
                );
            }
        }
    }
    // Rank two: x1+x3=4, x2+x3=5 has the minimum-norm solution [1,2,3].
    let a = Matrix::from_vec(
        4,
        3,
        [1., 0., 1., 0., 1., 1., 1., 1., 2., 0., 0., 0.]
            .map(real::<S>)
            .to_vec(),
    )
    .unwrap();
    let b = Vector::from_vec([4., 5., 9., 2.].map(real::<S>).to_vec());
    for result in least_squares(&a.view().try_finite().unwrap(), &b) {
        check(&a, &b, &result);
        for (&value, expected) in result.solution().as_slice().iter().zip([1., 2., 3.]) {
            assert!(
                (value - real::<S>(expected)).abs()
                    < S::Real::SQRT_EPSILON * S::Real::try_from_f64_lossy(20.).unwrap()
            );
        }
    }
    let a = Matrix::from_vec(2, 3, [1., 0., 1., 0., 1., 1.].map(real::<S>).to_vec()).unwrap();
    let b = Vector::from_vec([4., 5.].map(real::<S>).to_vec());
    for result in least_squares(&a.view().try_finite().unwrap(), &b) {
        check(&a, &b, &result);
    }
    // Singular Hermitian least squares: the null row supplies an orthogonal residual.
    let singular = Matrix::from_vec(
        3,
        3,
        [1., 0., 0., 0., 0., 0., 0., 0., -2.]
            .map(real::<S>)
            .to_vec(),
    )
    .unwrap();
    let singular_rhs = Vector::from_vec([1., 3., -2.].map(real::<S>).to_vec());
    let singular_result = MinresSolver::new(tolerance::<S>(), budget())
        .with_normal_tolerance(normal::<S>())
        .solve_with_options(
            &singular.view().try_finite().unwrap(),
            &singular_rhs,
            Preconditioning::None,
            None,
        )
        .unwrap();
    check(&singular, &singular_rhs, &singular_result);
    assert_eq!(
        singular_result.report().termination(),
        LinearSolveTermination::LeastSquaresConverged
    );
    // Hermitian indefinite, with genuinely complex conjugate off-diagonals.
    let a = Matrix::from_vec(
        3,
        3,
        vec![
            real::<S>(-2.),
            phase,
            real::<S>(0.),
            phase.conj(),
            real::<S>(3.),
            real::<S>(1.),
            real::<S>(0.),
            real::<S>(1.),
            real::<S>(-1.),
        ],
    )
    .unwrap();
    let b = Vector::from_vec(vec![
        real::<S>(-2.) + phase * real::<S>(2.),
        phase.conj() + real::<S>(9.),
        real::<S>(-1.),
    ]);
    let result = MinresSolver::new(tolerance::<S>(), budget())
        .with_normal_tolerance(normal::<S>())
        .solve_with_options(
            &a.view().try_finite().unwrap(),
            &b,
            Preconditioning::None,
            None,
        )
        .unwrap();
    check(&a, &b, &result);
    let csr = a
        .to_sparse_csr(scioxide_sparse::ExplicitZeroPolicy::Drop)
        .unwrap();
    let csc = csr.to_csc().unwrap();
    let view = a.view();
    let operator = IndependentOperator(&a);
    for op in [
        &view.try_finite().unwrap() as &dyn scioxide_core::LinearOperator<S>,
        &csr,
        &csc,
        &operator,
    ] {
        let result = MinresSolver::new(tolerance::<S>(), budget())
            .with_normal_tolerance(normal::<S>())
            .solve_with_options(op, &b, Preconditioning::None, None)
            .unwrap();
        check(&a, &b, &result);
    }
}
#[test]
fn exact_f32() {
    exact_fixtures::<f32>(1.);
}
#[test]
fn exact_f64() {
    exact_fixtures::<f64>(1.);
}
#[test]
fn exact_complex_f32() {
    exact_fixtures(Complex::new(0.6_f32, 0.8));
}
#[test]
fn exact_complex_f64() {
    exact_fixtures(Complex::new(0.6_f64, 0.8));
}

#[test]
fn empty_and_zero_operators() {
    for (m, n) in [(0, 0), (0, 3), (3, 0), (3, 2)] {
        let a = Matrix::<f64>::zeros(m, n).unwrap();
        let b = Vector::from_vec(vec![1.; m]);
        for result in least_squares(&a.view().try_finite().unwrap(), &b) {
            check(&a, &b, &result);
            assert_eq!(result.report().iterations(), 0);
        }
    }
    let a = Matrix::<f64>::zeros(0, 0).unwrap();
    let b = Vector::from_vec(vec![]);
    assert!(
        MinresSolver::new(tolerance::<f64>(), budget())
            .solve_with_options(
                &a.view().try_finite().unwrap(),
                &b,
                Preconditioning::None,
                None
            )
            .unwrap()
            .is_success()
    );
}

fn checkpoint_fixture<S: NumericScalar>() {
    let n = 120;
    let mut values = vec![S::ZERO; n * n];
    for (i, row) in values.chunks_mut(n).enumerate() {
        row[i] = real::<S>((i + 1) as f64);
    }
    let a = Matrix::from_vec(n, n, values).unwrap();
    let b = Vector::from_vec(vec![S::ONE; n]);
    let tolerance = ResidualTolerance::new(S::Real::ZERO, S::Real::ZERO).unwrap();
    let gradient = GradientTolerance::new(S::Real::ZERO, S::Real::ZERO).unwrap();
    let budget = Budget::try_from(65).unwrap();
    let reports = [
        MinresSolver::new(tolerance, budget)
            .with_normal_tolerance(gradient)
            .with_history(true)
            .solve_with_options(
                &a.view().try_finite().unwrap(),
                &b,
                Preconditioning::None,
                None,
            )
            .unwrap(),
        LsqrSolver::new(tolerance, budget)
            .with_normal_tolerance(gradient)
            .with_history(true)
            .solve_with_options(&a.view().try_finite().unwrap(), &b, None)
            .unwrap(),
        LsmrSolver::new(tolerance, budget)
            .with_normal_tolerance(gradient)
            .with_history(true)
            .solve_with_options(&a.view().try_finite().unwrap(), &b, None)
            .unwrap(),
    ];
    for report in reports {
        assert!(!report.is_success());
        assert_eq!(report.report().krylov().unwrap().restarts(), 0);
        assert!(
            report.report().true_residual().unwrap().raw_norm() < real::<S>(11.).real_component()
        );
        assert_eq!(
            report.history().unwrap().len(),
            report.report().iterations() + 1
        );
        // A fresh independent operator application must agree with the final report.
        let residual = a
            .view()
            .mul_vec(report.solution())
            .unwrap()
            .view()
            .try_sub(&b)
            .unwrap()
            .view()
            .norm2_scoped(
                &scioxide_linalg::LinalgExecutionOptions::new(),
                scioxide_core::ExecutionContext::sequential(),
            )
            .unwrap();
        assert!(
            (residual - report.report().true_residual().unwrap().raw_norm()).abs()
                <= S::Real::SQRT_EPSILON
        );
    }
}
#[test]
fn verification_preserves_recurrence_f32() {
    checkpoint_fixture::<f32>();
}
#[test]
fn verification_preserves_recurrence_f64() {
    checkpoint_fixture::<f64>();
}

#[test]
fn condition_limit_and_initial_nullspace_are_explicit() {
    use scioxide_linalg::ReciprocalConditionThreshold;
    let a = Matrix::from_vec(3, 3, vec![1., 0., 0., 0., 10., 0., 0., 0., 100.]).unwrap();
    let b = Vector::from_vec(vec![1.; 3]);
    let tol = ResidualTolerance::new(0., 0.).unwrap();
    let grad = GradientTolerance::new(0., 0.).unwrap();
    let threshold = ReciprocalConditionThreshold::new(0.99).unwrap();
    for result in [
        MinresSolver::new(tol, budget())
            .with_normal_tolerance(grad)
            .with_condition_limit(threshold)
            .solve_with_options(
                &a.view().try_finite().unwrap(),
                &b,
                Preconditioning::None,
                None,
            )
            .unwrap(),
        LsqrSolver::new(tol, budget())
            .with_normal_tolerance(grad)
            .with_condition_limit(threshold)
            .solve_with_options(&a.view().try_finite().unwrap(), &b, None)
            .unwrap(),
        LsmrSolver::new(tol, budget())
            .with_normal_tolerance(grad)
            .with_condition_limit(threshold)
            .solve_with_options(&a.view().try_finite().unwrap(), &b, None)
            .unwrap(),
    ] {
        assert_eq!(
            result.report().termination(),
            LinearSolveTermination::ConditionLimit
        );
    }
    let a = Matrix::from_vec(2, 3, vec![1., 0., 0., 0., 1., 0.]).unwrap();
    let b = Vector::from_vec(vec![1., 2.]);
    let initial = Vector::from_vec(vec![0., 0., 7.]);
    for result in [
        LsqrSolver::new(tolerance::<f64>(), budget())
            .solve_with_options(&a.view().try_finite().unwrap(), &b, Some(initial.view()))
            .unwrap(),
        LsmrSolver::new(tolerance::<f64>(), budget())
            .solve_with_options(&a.view().try_finite().unwrap(), &b, Some(initial.view()))
            .unwrap(),
    ] {
        assert!(result.is_success());
        assert_eq!(result.solution().as_slice()[2], 7.);
    }
}

// Independent mathematical provider used for real/complex matrix-free equivalence.
struct IndependentOperator<'a, S: NumericScalar>(&'a Matrix<S>);
impl<S: NumericScalar> scioxide_core::LinearOperator<S> for IndependentOperator<'_, S> {
    fn rows(&self) -> usize {
        self.0.rows()
    }
    fn cols(&self) -> usize {
        self.0.cols()
    }
    fn apply_workspace_len(&self) -> Result<usize, scioxide_core::SciOxideError> {
        Ok(0)
    }
    fn apply_into(
        &self,
        x: &[S],
        y: &mut [S],
        _: &mut [S],
    ) -> Result<(), scioxide_core::SciOxideError> {
        for (row, y) in y.iter_mut().enumerate() {
            *y = x.iter().enumerate().fold(S::ZERO, |sum, (col, &x)| {
                sum + self.0.view().get(row, col).unwrap() * x
            });
        }
        Ok(())
    }
}
impl<S: NumericScalar> AdjointOperator<S> for IndependentOperator<'_, S> {
    fn apply_adjoint_workspace_len(&self) -> Result<usize, scioxide_core::SciOxideError> {
        Ok(0)
    }
    fn apply_adjoint_into(
        &self,
        x: &[S],
        y: &mut [S],
        _: &mut [S],
    ) -> Result<(), scioxide_core::SciOxideError> {
        for (col, y) in y.iter_mut().enumerate() {
            *y = x.iter().enumerate().fold(S::ZERO, |sum, (row, &x)| {
                sum + self.0.view().get(row, col).unwrap().conj() * x
            });
        }
        Ok(())
    }
}

#[test]
fn successful_short_recurrences_convert_to_nested_solutions() {
    // Hermitian MINRES system: diag(2, 4) x = (2, 8) has x = (1, 2).
    let square = Matrix::from_vec(2, 2, vec![2., 0., 0., 4.]).unwrap();
    let rhs = Vector::from_vec(vec![2., 8.]);
    let square = square.view().try_finite().unwrap();
    let minres = MinresSolver::new(tolerance::<f64>(), budget())
        .solve_with_options(&square, &rhs, Preconditioning::None, None)
        .unwrap();
    assert_eq!(
        minres.report().termination(),
        LinearSolveTermination::Converged
    );
    let x = minres.into_converged_solution().unwrap();
    assert!((x.as_slice()[0] - 1.).abs() < 1e-12 && (x.as_slice()[1] - 2.).abs() < 1e-12);
    // Inconsistent tall system: the least-squares solution is (1, 2) with residual (0, 0, 3).
    let tall = Matrix::from_vec(3, 2, vec![1., 0., 0., 1., 0., 0.]).unwrap();
    let rhs = Vector::from_vec(vec![1., 2., 3.]);
    for result in least_squares(&tall.view().try_finite().unwrap(), &rhs) {
        assert_eq!(
            result.report().termination(),
            LinearSolveTermination::LeastSquaresConverged
        );
        assert!(result.is_success());
        let x = result.into_converged_solution().unwrap();
        assert!((x.as_slice()[0] - 1.).abs() < 1e-12 && (x.as_slice()[1] - 2.).abs() < 1e-12);
    }
}

use crate::support::krylov_reference;

#[test]
fn long_recurrences_match_unrestarted_reference_counts() {
    let tolerance = ResidualTolerance::new(0., 1e-8).unwrap();
    let limit = Budget::try_from(20000).unwrap();
    for fixture in krylov_reference::fixtures() {
        let b = fixture.rhs();
        let result = match fixture.method {
            "lsqr" => LsqrSolver::new(tolerance, limit)
                .solve_with_options(&fixture, &b, None)
                .unwrap(),
            "lsmr" => LsmrSolver::new(tolerance, limit)
                .solve_with_options(&fixture, &b, None)
                .unwrap(),
            _ => continue,
        };
        fixture.check(&result);
    }
}

#[test]
fn minres_keeps_a_bounded_least_squares_solution_of_singular_inconsistent_systems() {
    // `A = Q diag(d) Q` with a Householder reflector `Q`; `b` has components in
    // the null space, so `‖b − A x‖ ≥ ‖P_N b‖` and a pivot of the projected
    // tridiagonal reaches the rounding level once the range is resolved.
    for diagonal in [
        vec![2., 0.],
        vec![1., 2., 3., 0., 0.],
        vec![1., 2., -3., 4., 5., 6., 0., 0.],
    ] {
        let n = diagonal.len();
        let v: Vec<f64> = (0..n).map(|i| 1. + 0.37 * i as f64).collect();
        let vv: f64 = v.iter().map(|x| x * x).sum();
        let q = |i: usize, j: usize| f64::from(u8::from(i == j)) - 2. * v[i] * v[j] / vv;
        let entry =
            |i: usize, j: usize| -> f64 { (0..n).map(|k| q(i, k) * diagonal[k] * q(k, j)).sum() };
        let mut entries = vec![0.; n * n];
        for i in 0..n {
            for j in 0..=i {
                let value = 0.5 * (entry(i, j) + entry(j, i));
                entries[i * n + j] = value;
                entries[j * n + i] = value;
            }
        }
        let a = Matrix::from_vec(n, n, entries.clone()).unwrap();
        let operator = a.view().try_finite().unwrap();
        let b = vec![1.; n];
        // Minimum-norm least-squares solution `Q D⁺ Q b`.
        let qb: Vec<f64> = (0..n)
            .map(|i| (0..n).map(|j| q(i, j)).sum::<f64>())
            .collect();
        let minimum: f64 = (0..n)
            .filter(|&k| diagonal[k] != 0.)
            .map(|k| (qb[k] / diagonal[k]).powi(2))
            .sum::<f64>()
            .sqrt();
        let null_residual: f64 = (0..n)
            .filter(|&k| diagonal[k] == 0.)
            .map(|k| qb[k] * qb[k])
            .sum::<f64>()
            .sqrt();
        let tolerance = ResidualTolerance::new(0., 1e-10).unwrap();
        for limit in (1..=2 * n).chain([100]) {
            let result = MinresSolver::new(tolerance, Budget::try_from(limit).unwrap())
                .solve_with_options(&operator, &b[..], Preconditioning::None, None)
                .unwrap();
            let x = result.solution().as_slice();
            let norm = x.iter().fold(0_f64, |s, &v| s.hypot(v));
            let residual: Vec<f64> = (0..n)
                .map(|i| b[i] - (0..n).map(|j| entries[i * n + j] * x[j]).sum::<f64>())
                .collect();
            let normal = (0..n)
                .map(|i| {
                    (0..n)
                        .map(|j| entries[i * n + j] * residual[j])
                        .sum::<f64>()
                })
                .fold(0_f64, f64::hypot);
            let label = format!("{diagonal:?} limit={limit}: {:?}", result.report());
            assert!(norm <= 20. * minimum, "{label}: ‖x‖ = {norm:e}");
            if limit == 100 {
                assert_eq!(
                    result.report().termination(),
                    LinearSolveTermination::LeastSquaresConverged,
                    "{label}"
                );
            }
            if result.report().termination() == LinearSolveTermination::LeastSquaresConverged {
                assert!(normal <= 1e-12, "{label}: ‖A r‖ = {normal:e}");
                let r = residual.iter().fold(0_f64, |s, &v| s.hypot(v));
                assert!(
                    (r - null_residual).abs() <= 1e-12 * null_residual,
                    "{label}"
                );
            }
        }
        if n == 2 {
            // Without a usable normal-residual test the rounding-level pivot is
            // rank loss: the update is skipped, the restart sees only rounding
            // noise, and the solve stops at the least-squares iterate.
            let result = MinresSolver::new(tolerance, Budget::try_from(100).unwrap())
                .with_normal_tolerance(GradientTolerance::new(0., 1e-300).unwrap())
                .solve_with_options(&operator, &b[..], Preconditioning::None, None)
                .unwrap();
            let report = result.report();
            assert_eq!(report.termination(), LinearSolveTermination::Stagnation);
            assert!(report.krylov().unwrap().rank_loss_detected());
            let x = result.solution().as_slice();
            assert!(
                x.iter().fold(0_f64, |s, &v| s.hypot(v)) <= 20. * minimum,
                "{x:?}"
            );
            let r = (0..n)
                .map(|i| b[i] - (0..n).map(|j| entries[i * n + j] * x[j]).sum::<f64>())
                .fold(0_f64, f64::hypot);
            assert!((r - null_residual).abs() <= 1e-12 * null_residual, "{r:e}");
        }
    }
}
