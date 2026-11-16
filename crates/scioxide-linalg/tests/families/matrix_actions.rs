//! Sparse-compatible analytic exponential/phi actions and operator contracts.
#![allow(clippy::unwrap_used)]

use scioxide_core::AlgebraicScalar;
use scioxide_core::{
    BlockVectorView, BlockVectorViewMut, Budget, Complex, ExecutionContext, NumericScalar,
    RealScalar,
};
use scioxide_linalg::{ExponentialAction, Matrix, OneNormEstimator, SpectralTermination};
#[path = "../matrix_actions/budgets.rs"]
mod budgets;
#[path = "../matrix_actions/edges.rs"]
mod edges;
#[path = "../matrix_actions/efficiency.rs"]
mod efficiency;
#[path = "../matrix_actions/owned.rs"]
mod owned;
#[path = "../matrix_actions/references.rs"]
mod references;
#[path = "../matrix_actions/shifts.rs"]
mod shifts;
#[path = "../matrix_actions/stops.rs"]
mod stops;
fn run<S: NumericScalar>() {
    let v = |x| S::from_real(S::Real::try_from_f64_lossy(x).unwrap());
    let a = Matrix::from_vec(2, 2, vec![v(0.), v(1.), v(0.), v(0.)]).unwrap();
    let options = ExponentialAction::new(S::Real::ONE).with_max_phi(3);
    let p = options;
    let mut w = p.workspace(&a.view().try_finite().unwrap(), 2).unwrap();
    let b = vec![v(1.), v(2.), v(0.), v(0.)];
    let mut out = vec![S::ZERO; 4];
    let tol = S::Real::try_from_usize_exact(256).unwrap() * <S::Real as RealScalar>::EPSILON;
    let factorial = [1., 1., 2., 6., 24.];
    for phi in 0..=3 {
        let report = p
            .apply_into(
                &a.view().try_finite().unwrap(),
                scioxide_linalg::PhiFunction::phi(phi, S::Real::ONE),
                BlockVectorView::new(&b, 2, 2).unwrap(),
                BlockVectorViewMut::new(&mut out, 2, 2).unwrap(),
                &mut w,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert_eq!(report.termination(), SpectralTermination::Converged);
        assert!((out[0] - v(1. / factorial[phi] + 2. / factorial[phi + 1])).abs() < tol);
        assert!((out[1] - v(2. / factorial[phi])).abs() < tol);
        assert_eq!(&out[2..], &[S::ZERO, S::ZERO]);
    }
    let norm = OneNormEstimator::new().with_applications(Budget::ONE.saturating_add(19));
    let mut nw = norm.workspace(&a.view().try_finite().unwrap()).unwrap();
    let report = norm
        .estimate_scoped(
            &a.view().try_finite().unwrap(),
            &mut nw,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(report.estimate(), S::Real::ONE);
    assert_eq!(
        norm.estimate_scoped(
            &a.view().try_finite().unwrap(),
            &mut nw,
            ExecutionContext::sequential()
        )
        .unwrap()
        .estimate(),
        S::Real::ONE
    );
}
#[test]
fn real32() {
    run::<f32>()
}
#[test]
fn real64() {
    run::<f64>()
}
#[test]
fn complex32() {
    run::<Complex<f32>>()
}
#[test]
fn complex64() {
    run::<Complex<f64>>()
}

/// Q D Q^T, Q = I - 2 u u^T, with a known clustered positive spectrum.
/// The analytic product avoids cubic fixture generation or dense test storage.
struct Clustered {
    u: Vec<f64>,
    diagonal: Vec<f64>,
    weighted: f64,
    calls: core::cell::Cell<usize>,
}
impl Clustered {
    fn new(n: usize) -> Self {
        let length = (1..=n).map(|i| (i as f64).powi(2)).sum::<f64>().sqrt();
        let u: Vec<_> = (1..=n).map(|i| i as f64 / length).collect();
        let diagonal: Vec<_> = (0..n).map(|i| 1.0 + i as f64 / n as f64).collect();
        let weighted = u.iter().zip(&diagonal).map(|(u, d)| u * u * d).sum();
        Self {
            u,
            diagonal,
            weighted,
            calls: core::cell::Cell::new(0),
        }
    }
}
impl scioxide_core::LinearOperator<f64> for Clustered {
    fn rows(&self) -> usize {
        self.u.len()
    }
    fn cols(&self) -> usize {
        self.u.len()
    }
    fn apply_workspace_len(&self) -> Result<usize, scioxide_core::SciOxideError> {
        Ok(0)
    }
    fn apply_into(
        &self,
        x: &[f64],
        y: &mut [f64],
        _: &mut [f64],
    ) -> Result<(), scioxide_core::SciOxideError> {
        self.calls.set(self.calls.get() + 1);
        let ux: f64 = self.u.iter().zip(x).map(|(u, x)| u * x).sum();
        let udx: f64 = self
            .u
            .iter()
            .zip(&self.diagonal)
            .zip(x)
            .map(|((u, d), x)| u * d * x)
            .sum();
        for (((y, &x), &u), &d) in y.iter_mut().zip(x).zip(&self.u).zip(&self.diagonal) {
            *y = d * x - 2.0 * u * (udx + d * ux - 2.0 * self.weighted * ux);
        }
        Ok(())
    }
}
impl scioxide_core::AdjointOperator<f64> for Clustered {
    fn apply_adjoint_workspace_len(&self) -> Result<usize, scioxide_core::SciOxideError> {
        Ok(0)
    }
    fn apply_adjoint_into(
        &self,
        x: &[f64],
        y: &mut [f64],
        w: &mut [f64],
    ) -> Result<(), scioxide_core::SciOxideError> {
        scioxide_core::LinearOperator::apply_into(self, x, y, w)
    }
}

#[test]
fn clustered_leading_spectra_converge_at_1024_with_default_cycles() {
    use scioxide_core::LinearOperator;
    use scioxide_linalg::{
        OperatorStructure, PartialSpectrumBuffer, PartialSpectrumSolver, PartialSvdBuffer,
        PartialSvdSolver,
    };
    let n = 1024;
    let count = 6;
    let a = Clustered::new(n);
    let initial = vec![1.0; n];
    let options = PartialSpectrumSolver::new(count).with_structure(OperatorStructure::Hermitian);
    let plan = options;
    let mut workspace = plan.workspace(&a).unwrap();
    let mut result = PartialSpectrumBuffer::new(n, count).unwrap();
    let result_report = plan
        .solve_into(
            &a,
            initial.as_slice(),
            &mut result,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(result_report.termination(), SpectralTermination::Converged);
    assert!(result_report.cycles() > 1 && result_report.cycles() <= 100);
    assert_eq!(result_report.applications(), a.calls.get());
    let mut vector = vec![0.; n];
    let mut image = vec![0.; n];
    for (j, lambda) in result.eigenvalues().iter().enumerate() {
        assert!((lambda.re - a.diagonal[n - 1 - j]).abs() < 1e-8);
        assert!(lambda.im.abs() < 1e-14);
        for (i, x) in vector.iter_mut().enumerate() {
            *x = result.eigenvectors().get(i, j).unwrap().re;
        }
        a.apply_into(&vector, &mut image, &mut []).unwrap();
        let residual = image
            .iter()
            .zip(&vector)
            .map(|(a, x)| (a - lambda.re * x).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(residual <= options.tolerance().threshold(lambda.re).unwrap());
        assert!((residual - result.residuals()[j]).abs() < 1e-13);
        for k in 0..=j {
            let dot: f64 = vector
                .iter()
                .enumerate()
                .map(|(i, x)| x * result.eigenvectors().get(i, k).unwrap().re)
                .sum();
            assert!((dot - if j == k { 1. } else { 0. }).abs() < 1e-8);
        }
    }
    a.calls.set(0);
    let options = PartialSvdSolver::new(count);
    let plan = options;
    let mut workspace = plan.workspace(&a).unwrap();
    let mut result = PartialSvdBuffer::new(n, n, count).unwrap();
    let result_report = plan
        .solve_into(
            &a,
            &initial,
            &mut result,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(result_report.termination(), SpectralTermination::Converged);
    assert!(result_report.cycles() > 1 && result_report.cycles() <= 100);
    assert_eq!(result_report.applications(), a.calls.get());
    for (j, &sigma) in result.singular_values().iter().enumerate() {
        assert!((sigma - a.diagonal[n - 1 - j]).abs() < 1e-8);
        for (side, (vectors, partners)) in [
            (result.right_vectors(), result.left_vectors()),
            (result.left_vectors(), result.right_vectors()),
        ]
        .into_iter()
        .enumerate()
        {
            for (i, x) in vector.iter_mut().enumerate() {
                *x = vectors.get(i, j).unwrap();
            }
            a.apply_into(&vector, &mut image, &mut []).unwrap();
            let residual = image
                .iter()
                .enumerate()
                .map(|(i, a)| (a - sigma * partners.get(i, j).unwrap()).powi(2))
                .sum::<f64>()
                .sqrt();
            assert!(residual <= options.tolerance().threshold(sigma).unwrap());
            assert!((residual - result.residuals()[j][side]).abs() < 1e-13);
            for k in 0..=j {
                let dot: f64 = vector
                    .iter()
                    .enumerate()
                    .map(|(i, x)| x * vectors.get(i, k).unwrap())
                    .sum();
                assert!((dot - if j == k { 1. } else { 0. }).abs() < 1e-8);
            }
        }
    }
}

#[test]
fn adaptive_exponential_large_nonnormal_norm_and_negative_time() {
    // A^2=I, despite ||A||_1=1001. Power estimates avoid overscaling.
    let a = Matrix::from_vec(2, 2, vec![1.0_f64, 1000., 0., -1.]).unwrap();
    let operator = a.view().try_finite().unwrap();
    let plan = ExponentialAction::new(1001.)
        .with_tolerance(scioxide_core::Tolerance::new(0., f64::EPSILON).unwrap());
    let mut workspace = plan.workspace(&operator, 1).unwrap();
    for t in [-1.0_f64, 1.] {
        let mut out = [0.; 2];
        let report = plan
            .apply_into(
                &operator,
                scioxide_linalg::PhiFunction::phi(0, t),
                BlockVectorView::new(&[0., 1.], 2, 1).unwrap(),
                BlockVectorViewMut::new(&mut out, 2, 1).unwrap(),
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert_eq!(report.termination(), SpectralTermination::Converged);
        assert!(report.applications() < 200);
        assert!((out[0] / (1000. * t.sinh()) - 1.).abs() < 1e-13);
        assert!((out[1] / (-t).exp() - 1.).abs() < 1e-13);
    }
}

#[test]
fn adaptive_phi_negative_time_and_work_limits() {
    let a = Matrix::from_vec(2, 2, vec![0.0_f64, 1000., 0., 0.]).unwrap();
    let operator = a.view().try_finite().unwrap();
    let options = ExponentialAction::new(1000.)
        .with_max_phi(3)
        .with_tolerance(scioxide_core::Tolerance::new(0., f64::EPSILON).unwrap());
    let plan = options;
    let mut workspace = plan.workspace(&operator, 1).unwrap();
    let factorial = [1., 1., 2., 6., 24.];
    for phi in 1..=3 {
        let mut out = [0.; 2];
        let report = plan
            .apply_into(
                &operator,
                scioxide_linalg::PhiFunction::phi(phi, -1.),
                BlockVectorView::new(&[1., 2.], 2, 1).unwrap(),
                BlockVectorViewMut::new(&mut out, 2, 1).unwrap(),
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert_eq!(report.termination(), SpectralTermination::Converged);
        assert!((out[0] - (1. / factorial[phi] - 2000. / factorial[phi + 1])).abs() < 1e-11);
        assert!((out[1] - 2. / factorial[phi]).abs() < 1e-13);
    }
    // Stop inside power estimation (large bound), and inside the Taylor
    // recurrence (small bound). Neither incomplete stage is published.
    for bound in [1., 1000.] {
        let a = Matrix::from_vec(2, 2, vec![-1., 0., 0., -1.]).unwrap();
        let operator = a.view().try_finite().unwrap();
        let plan = ExponentialAction::new(bound).with_applications(Budget::ONE);
        let mut workspace = plan.workspace(&operator, 1).unwrap();
        let mut out = [42.; 2];
        let report = plan
            .apply_into(
                &operator,
                scioxide_linalg::PhiFunction::phi(0, 1.),
                BlockVectorView::new(&[1., 2.], 2, 1).unwrap(),
                BlockVectorViewMut::new(&mut out, 2, 1).unwrap(),
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert_eq!(report.termination(), SpectralTermination::EvaluationLimit);
        assert_eq!(report.applications(), 1);
        assert_eq!(workspace.last_accepted().unwrap().progress(), &[0.]);
        assert_eq!(workspace.last_accepted().unwrap().completed(), &[false]);
        assert_eq!(out, [1., 2.]);
    }
}

#[test]
fn augmented_power_estimation_uses_complex_adjoints() {
    let mut a = Matrix::<Complex<f64>>::zeros(6, 6).unwrap();
    a.view_mut().set(0, 1, Complex::new(0., 1000.)).unwrap();
    a.view_mut().set(2, 3, Complex::new(0., -400.)).unwrap();
    let operator = a.view().try_finite().unwrap();
    let b = [Complex::new(1., 2.); 6];
    let options = ExponentialAction::new(1000.)
        .with_max_phi(2)
        .with_tolerance(scioxide_core::Tolerance::new(0., f64::EPSILON).unwrap());
    let plan = options;
    let mut workspace = plan.workspace(&operator, 1).unwrap();
    let factorial = [1., 1., 2., 6.];
    for phi in 0..=2 {
        let mut out = [Complex::ZERO; 6];
        let report = plan
            .apply_into(
                &operator,
                scioxide_linalg::PhiFunction::phi(phi, -0.5),
                BlockVectorView::new(&b, 6, 1).unwrap(),
                BlockVectorViewMut::new(&mut out, 6, 1).unwrap(),
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert_eq!(report.termination(), SpectralTermination::Converged);
        // A^2=0 gives an independent two-term reference for every phi index.
        for (i, value) in out.iter().enumerate() {
            let image = match i {
                0 => Complex::new(0., 1000.) * b[1],
                2 => Complex::new(0., -400.) * b[3],
                _ => Complex::ZERO,
            };
            let expected = b[i] / Complex::from_real(factorial[phi])
                - image * Complex::from_real(0.5 / factorial[phi + 1]);
            assert!((*value - expected).norm() < 1e-10);
        }
    }
}

#[test]
fn thick_restart_complex_rectangular_svd_and_budget_stops() {
    use scioxide_linalg::{PartialSvdBuffer, PartialSvdSolver};
    let (rows, cols, count) = (48, 40, 4);
    let mut a = Matrix::<Complex<f64>>::zeros(rows, cols).unwrap();
    for j in 0..cols {
        let sigma = 1. + j as f64 / cols as f64;
        a.view_mut()
            .set(
                j,
                j,
                if j % 2 == 0 {
                    Complex::new(0., sigma)
                } else {
                    Complex::new(-sigma, 0.)
                },
            )
            .unwrap();
    }
    let operator = a.view().try_finite().unwrap();
    let options = PartialSvdSolver::new(count).with_basis(Budget::try_from(12).unwrap());
    let plan = options;
    let mut workspace = plan.workspace(&operator).unwrap();
    let mut result = PartialSvdBuffer::new(rows, cols, count).unwrap();
    let initial = vec![Complex::new(1., 0.5); cols];
    let result_report = plan
        .solve_into(
            &operator,
            &initial,
            &mut result,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(result_report.termination(), SpectralTermination::Converged);
    assert!(result_report.cycles() > 1);
    for (j, &sigma) in result.singular_values().iter().enumerate() {
        assert!((sigma - (1. + (cols - 1 - j) as f64 / cols as f64)).abs() < 1e-8);
        assert!(
            result.residuals()[j]
                .iter()
                .all(|&r| r <= options.tolerance().threshold(sigma).unwrap())
        );
    }
    for (options, status) in [
        (
            options.with_applications(Budget::ONE),
            SpectralTermination::EvaluationLimit,
        ),
        (
            options.with_applications(Budget::try_from(40).unwrap()),
            SpectralTermination::EvaluationLimit,
        ),
        (
            options.with_execution_options(
                (options)
                    .execution_options()
                    .with_iteration_budget(Budget::ONE),
            ),
            SpectralTermination::IterationLimit,
        ),
    ] {
        let plan = options;
        let result_report = plan
            .solve_into(
                &operator,
                &initial,
                &mut result,
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert_eq!(result_report.termination(), status);
        if options.applications().get() == 40 {
            // A stop during a later expansion retains the first independently
            // verified snapshot instead of publishing an unchecked projection.
            assert_eq!(result.singular_values().len(), count);
        }
    }
}

#[test]
fn repeated_values_exhaust_the_starting_invariant_subspace() {
    use scioxide_linalg::{
        OperatorStructure, PartialSpectrumSolver, PartialSvdBuffer, PartialSvdSolver,
    };
    let mut a = Matrix::<f64>::zeros(6, 6).unwrap();
    for (i, value) in [3., 3., 2., 2., 1., 1.].into_iter().enumerate() {
        a.view_mut().set(i, i, value).unwrap();
    }
    let operator = a.view().try_finite().unwrap();
    let plan = PartialSpectrumSolver::new(4).with_structure(OperatorStructure::Hermitian);
    let mut result = scioxide_linalg::PartialSpectrumBuffer::new(6, 4).unwrap();
    let result_report = plan
        .solve_into(
            &operator,
            [1.; 6].as_slice(),
            &mut result,
            &mut plan.workspace(&operator).unwrap(),
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(
        result_report.termination(),
        SpectralTermination::InvariantSubspace
    );
    assert_eq!(result.eigenvalues().len(), 3);
    let plan = PartialSvdSolver::new(4);
    let mut workspace = plan.workspace(&operator).unwrap();
    let mut result = PartialSvdBuffer::new(6, 6, 4).unwrap();
    let result_report = plan
        .solve_into(
            &operator,
            &[1.; 6],
            &mut result,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(
        result_report.termination(),
        SpectralTermination::InvariantSubspace
    );
    assert_eq!(result.singular_values().len(), 3);
}

#[test]
fn exponential_work_limit_publishes_only_completed_scaling_steps() {
    let a = Matrix::from_vec(2, 2, vec![-20.0_f64, 0., 0., -10.]).unwrap();
    let operator = a.view().try_finite().unwrap();
    let plan = ExponentialAction::new(20.).with_applications(Budget::try_from(50).unwrap());
    let mut workspace = plan.workspace(&operator, 1).unwrap();
    let mut out = [42.; 2];
    let report = plan
        .apply_into(
            &operator,
            scioxide_linalg::PhiFunction::phi(0, 1.),
            BlockVectorView::new(&[1., 1.], 2, 1).unwrap(),
            BlockVectorViewMut::new(&mut out, 2, 1).unwrap(),
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(report.termination(), SpectralTermination::EvaluationLimit);
    assert!(
        workspace.last_accepted().unwrap().progress()[0] > 0.
            && workspace.last_accepted().unwrap().progress()[0] < 1.
    );
    assert!(
        (out[0] - (-20. * workspace.last_accepted().unwrap().progress()[0]).exp()).abs() < 1e-9
    );
    assert!(
        (out[1] - (-10. * workspace.last_accepted().unwrap().progress()[0]).exp()).abs() < 1e-9
    );
}

#[test]
fn high_phi_retains_relative_accuracy_in_the_small_augmented_block() {
    let a = Matrix::from_vec(1, 1, vec![-1.0_f64]).unwrap();
    let operator = a.view().try_finite().unwrap();
    let plan = ExponentialAction::new(1.)
        .with_max_phi(20)
        .with_tolerance(scioxide_core::Tolerance::new(0., f64::EPSILON).unwrap());
    let mut workspace = plan.workspace(&operator, 1).unwrap();
    let mut out = [0.];
    let report = plan
        .apply_into(
            &operator,
            scioxide_linalg::PhiFunction::phi(20, 1.),
            BlockVectorView::new(&[1.], 1, 1).unwrap(),
            BlockVectorViewMut::new(&mut out, 1, 1).unwrap(),
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(report.termination(), SpectralTermination::Converged);
    let mut term = 1. / (1..=20).map(f64::from).product::<f64>();
    let mut expected = term;
    for j in 1..=40 {
        term /= -f64::from(20 + j);
        expected += term;
    }
    assert!((out[0] / expected - 1.).abs() < 2e-15);
}
#[test]
fn restarted_lanczos_and_general_arnoldi() {
    use scioxide_linalg::{PartialSpectrumBuffer, PartialSpectrumSolver};
    let n = 24;
    let mut a = Matrix::<f64>::zeros(n, n).unwrap();
    for i in 0..n {
        a.view_mut().set(i, i, (i + 1) as f64).unwrap();
    }
    let options = PartialSpectrumSolver::new(2)
        .with_basis(Budget::try_from(10).unwrap())
        .with_structure(scioxide_linalg::OperatorStructure::Hermitian);
    let p = options;
    let mut w = p.workspace(&a.view().try_finite().unwrap()).unwrap();
    let mut out = PartialSpectrumBuffer::new(n, 2).unwrap();
    let initial = vec![1.; n];
    let out_report = p
        .solve_into(
            &a.view().try_finite().unwrap(),
            initial.as_slice(),
            &mut out,
            &mut w,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(
        out_report.termination(),
        SpectralTermination::Converged,
        "{out:?}"
    );
    assert!(out_report.cycles() > 1);
    assert!((out.eigenvalues()[0].re - 24.).abs() < 1e-5);
    assert!((out.eigenvalues()[1].re - 23.).abs() < 1e-5);
    let a = Matrix::from_vec(3, 3, vec![0.0_f64, -2., 1., 2., 0., 0., 0., 0., 1.]).unwrap();
    let p = PartialSpectrumSolver::new(2);
    let op = a.view().try_finite().unwrap();
    let mut result = PartialSpectrumBuffer::new(3, 2).unwrap();
    let result_report = p
        .solve_into(
            &op,
            [1., 0.5, 0.2].as_slice(),
            &mut result,
            &mut p.workspace(&op).unwrap(),
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(result_report.termination(), SpectralTermination::Converged);
    assert!((result.eigenvalues()[0].norm() - 2.).abs() < 1e-8);
}
#[test]
fn constrained_generalized_lobpcg() {
    use scioxide_linalg::{LobpcgBuffer, LobpcgSolver};
    let n = 12;
    let k = 2;
    let mut a = Matrix::<f64>::zeros(n, n).unwrap();
    let mut b = Matrix::<f64>::zeros(n, n).unwrap();
    for i in 0..n {
        a.view_mut().set(i, i, (i + 1) as f64).unwrap();
        b.view_mut().set(i, i, 2.).unwrap();
    }
    let plan = LobpcgSolver::new(k);
    let (a, b) = (
        a.view().try_finite().unwrap(),
        b.view().try_finite().unwrap(),
    );
    let pencil = scioxide_linalg::LobpcgOperator::new(&a).with_mass(&b);
    let mut w = plan.workspace(pencil, 1).unwrap();
    let mut initial = vec![1.; n * k];
    for i in 0..n {
        initial[n + i] = ((i * 7) % 11) as f64 - 4.;
    }
    let mut constraints = vec![0.; n];
    constraints[0] = 1.;
    let mut result = LobpcgBuffer::new(n, k).unwrap();
    let result_report = plan
        .solve_into(
            pencil,
            BlockVectorView::new(&initial, n, k).unwrap(),
            BlockVectorView::new(&constraints, n, 1).unwrap(),
            &mut result,
            &mut w,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(
        result_report.termination(),
        SpectralTermination::Converged,
        "{result:?}"
    );
    assert!((result.eigenvalues()[0] - 1.).abs() < 1e-7);
    assert!((result.eigenvalues()[1] - 1.5).abs() < 1e-7);
    assert!(result.eigenvectors().get(0, 0).unwrap().abs() < 1e-10);
}

/// `c · tridiag(-1 - skew, 2, -1)` of order 70 with the exact inverse of its
/// symmetric part as preconditioner.
fn scaled_laplacian(
    c: f64,
    skew: f64,
) -> (Matrix<f64>, scioxide_linalg::CholeskyDecomposition<f64>) {
    use scioxide_linalg::{CholeskyDecomposition, CholeskyPivoting, HermitianInput};
    let n = 70;
    let mut a = Matrix::<f64>::zeros(n, n).unwrap();
    let mut symmetric = Matrix::<f64>::zeros(n, n).unwrap();
    for i in 0..n {
        a.view_mut().set(i, i, 2. * c).unwrap();
        symmetric.view_mut().set(i, i, 2. * c).unwrap();
        if i + 1 < n {
            a.view_mut().set(i, i + 1, -c).unwrap();
            a.view_mut().set(i + 1, i, -c * (1. + skew)).unwrap();
            symmetric.view_mut().set(i, i + 1, -c).unwrap();
            symmetric.view_mut().set(i + 1, i, -c).unwrap();
        }
    }
    let factor =
        CholeskyDecomposition::factor(&symmetric, HermitianInput::Exact, CholeskyPivoting::None)
            .unwrap();
    (a, factor)
}

#[test]
fn lobpcg_hermitian_test_is_scale_invariant_and_independent_of_the_tolerance() {
    use scioxide_core::{ResidualTolerance, SciOxideErrorKind};
    use scioxide_linalg::{LobpcgOperator, LobpcgSolver};
    let (n, k) = (70, 4);
    let initial: Vec<f64> = (0..n * k)
        .map(|i| {
            let t = (i as f64 + 1.) * 0.618_033_988_7;
            t - t.floor() - 0.5
        })
        .collect();
    let initial = BlockVectorView::new(&initial, n, k).unwrap();
    let constraints = BlockVectorView::new(&[], n, 0).unwrap();
    let default = ResidualTolerance::from_tolerance(scioxide_core::Tolerance::from_sqrt_epsilon());
    // Exactly symmetric operators: large scales and tight tolerances once
    // amplified the rounding of the cached products past the test.
    for (c, tolerance) in [
        (1e3, default),
        (1e6, default),
        (1., ResidualTolerance::new(0., 1e-12).unwrap()),
        (1., ResidualTolerance::new(1e-12, 0.).unwrap()),
    ] {
        let (a, factor) = scaled_laplacian(c, 0.);
        let a = a.view().try_finite().unwrap();
        let inverse = factor.inverse_operator();
        let op = LobpcgOperator::new(&a).with_preconditioner(&inverse);
        let (output, report) = LobpcgSolver::new(k)
            .with_tolerance(tolerance)
            .solve(op, initial, constraints)
            .unwrap();
        assert_eq!(report.termination(), SpectralTermination::Converged);
        for (j, &lambda) in output.eigenvalues().iter().enumerate() {
            let angle = (j + 1) as f64 * core::f64::consts::PI / (2. * (n as f64 + 1.));
            let exact = 4. * c * angle.sin().powi(2);
            assert!(
                (lambda - exact).abs() <= 1e-9 * exact,
                "{c}: {lambda} vs {exact}"
            );
        }
    }
    // A skew part of relative size 1e-6 is still detected at any scale.
    for c in [1., 5041.] {
        let (a, factor) = scaled_laplacian(c, 1e-6);
        let a = a.view().try_finite().unwrap();
        let inverse = factor.inverse_operator();
        let op = LobpcgOperator::new(&a).with_preconditioner(&inverse);
        let error = LobpcgSolver::new(k)
            .solve(op, initial, constraints)
            .unwrap_err();
        assert_eq!(error.kind(), SciOxideErrorKind::InvalidInput, "{error}");
    }
}
#[test]
fn partial_svd_and_sketch_decompositions() {
    use scioxide_linalg::{
        InterpolativeBuffer, PartialSvdBuffer, PartialSvdSolver, RandomizedLowRank,
        RandomizedSvdBuffer,
    };
    let a = Matrix::from_vec(
        4,
        3,
        vec![3.0_f64, 0., 0., 0., 2., 0., 0., 0., 0., 0., 0., 0.],
    )
    .unwrap();
    let p = PartialSvdSolver::new(2);
    let mut w = p.workspace(&a.view().try_finite().unwrap()).unwrap();
    let mut out = PartialSvdBuffer::new(4, 3, 2).unwrap();
    let out_report = p
        .solve_into(
            &a.view().try_finite().unwrap(),
            &[1., 1., 1.],
            &mut out,
            &mut w,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(
        out_report.termination(),
        SpectralTermination::Converged,
        "{out:?}"
    );
    assert!((out.singular_values()[0] - 3.).abs() < 1e-10);
    assert!((out.singular_values()[1] - 2.).abs() < 1e-10);
    let sketch = Matrix::from_vec(3, 2, vec![1., 1., 1., -1., 1., 0.]).unwrap();
    let probes = Matrix::from_vec(3, 2, vec![1., 2., 3., -1., -2., 1.]).unwrap();
    let p = RandomizedLowRank::new(2);
    let mut w = p.workspace(&a.view().try_finite().unwrap(), 2).unwrap();
    let mut out = RandomizedSvdBuffer::new(4, 3, 2).unwrap();
    let out_report = p
        .svd_into(
            &a.view().try_finite().unwrap(),
            sketch.view(),
            probes.view(),
            &mut out,
            &mut w,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(out_report.termination(), SpectralTermination::Converged);
    assert!((out.singular_values()[0] - 3.).abs() < 1e-10);
    let mut id = InterpolativeBuffer::new(4, 3, 2).unwrap();
    let id_report = p
        .interpolative_into(
            &a.view().try_finite().unwrap(),
            sketch.view(),
            probes.view(),
            &mut id,
            &mut w,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(id_report.termination(), SpectralTermination::Converged);
    let restored = id.columns().mul_mat(id.coefficients()).unwrap();
    for (&x, &y) in restored.as_slice().iter().zip(a.as_slice()) {
        assert!((x - y).abs() < 1e-10);
    }
}
#[test]
fn restarted_general_function_action() {
    use scioxide_linalg::{FunctionAction, MatrixFunction};
    let n = 8;
    let mut a = Matrix::<f64>::zeros(n, n).unwrap();
    for i in 0..n {
        a.view_mut().set(i, i, 1. + i as f64 / 20.).unwrap();
    }
    let options = FunctionAction::default()
        .with_restart(Budget::try_from(4).unwrap())
        .with_projection(Budget::try_from(32).unwrap());
    let plan = options;
    let mut w = plan.workspace(&a.view().try_finite().unwrap(), 1).unwrap();
    let input = vec![1.; n];
    let mut out = vec![0.; n];
    let report = plan
        .apply_into(
            &a.view().try_finite().unwrap(),
            MatrixFunction::Sqrt,
            BlockVectorView::new(&input, n, 1).unwrap(),
            BlockVectorViewMut::new(&mut out, n, 1).unwrap(),
            &mut w,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(
        report.termination(),
        SpectralTermination::Converged,
        "{report:?}"
    );
    for (i, &v) in out.iter().enumerate() {
        assert!((v - (1. + i as f64 / 20.).sqrt()).abs() < 1e-6);
    }
}
#[test]
fn shift_invert_checks_original_pencil() {
    use scioxide_linalg::{PartialSpectrumBuffer, PartialSpectrumSolver, SpectrumOperator};
    let a = Matrix::from_vec(3, 3, vec![1.0_f64, 0., 0., 0., 3., 0., 0., 0., 5.]).unwrap();
    let a = a.view().try_finite().unwrap();
    let m = Matrix::<f64>::identity(3)
        .unwrap()
        .view()
        .try_scale(2.)
        .unwrap();
    let m = m.view().try_finite().unwrap();
    let inverse = Matrix::from_vec(3, 3, vec![-1., 0., 0., 0., 1., 0., 0., 0., 1. / 3.]).unwrap();
    let inverse = inverse.view().try_finite().unwrap();
    let operator = SpectrumOperator::ShiftInvert {
        a: &a,
        inverse: &inverse,
        mass: Some(&m),
        shift: 1.,
    };
    let p = PartialSpectrumSolver::new(2);
    let mut w = p.workspace(operator).unwrap();
    let mut result = PartialSpectrumBuffer::new(3, 2).unwrap();
    let result_report = p
        .solve_into(
            operator,
            [1., 2., 3.].as_slice(),
            &mut result,
            &mut w,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(result_report.termination(), SpectralTermination::Converged);
    for value in result.eigenvalues() {
        assert!((value.re - 0.5).abs() < 1e-10 || (value.re - 1.5).abs() < 1e-10);
    }
    // A direct-operator workspace lacks the verification storage.
    let mut direct = p.workspace(&a).unwrap();
    assert!(
        p.solve_into(
            operator,
            [1., 2., 3.].as_slice(),
            &mut result,
            &mut direct,
            ExecutionContext::sequential()
        )
        .is_err()
    );
    let bad = Matrix::<f64>::identity(3).unwrap();
    let bad = bad.view().try_finite().unwrap();
    let operator = SpectrumOperator::ShiftInvert {
        a: &a,
        inverse: &bad,
        mass: None,
        shift: 0.,
    };
    let p = PartialSpectrumSolver::new(1);
    let mut w = p.workspace(operator).unwrap();
    let mut result = PartialSpectrumBuffer::new(3, 1).unwrap();
    let result_report = p
        .solve_into(
            operator,
            [1., 2., 3.].as_slice(),
            &mut result,
            &mut w,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_ne!(result_report.termination(), SpectralTermination::Converged);
    assert!(result.residuals()[0] > 0.1);
    assert_eq!(
        w.last_attempt().unwrap().outcome(),
        scioxide_core::AttemptOutcome::Stopped
    );
}
fn all_spectral_domains<S: scioxide_core::NumericScalar>()
where
    Complex<S::Real>: NumericScalar<Real = S::Real>,
{
    use scioxide_linalg::{
        InterpolativeBuffer, LobpcgBuffer, LobpcgSolver, PartialSvdBuffer, PartialSvdSolver,
        RandomizedLowRank, RandomizedSvdBuffer,
    };
    let v = |r, i| {
        S::from_components(
            S::Real::try_from_f64_lossy(r).unwrap(),
            S::Real::try_from_f64_lossy(i).unwrap(),
        )
    };
    let phase = if S::IS_REAL { v(1., 0.) } else { v(0., 1.) };
    let a = Matrix::from_vec(
        4,
        3,
        vec![
            phase * v(3., 0.),
            S::ZERO,
            S::ZERO,
            S::ZERO,
            v(2., 0.),
            S::ZERO,
            S::ZERO,
            S::ZERO,
            S::ZERO,
            S::ZERO,
            S::ZERO,
            S::ZERO,
        ],
    )
    .unwrap();
    let p = PartialSvdSolver::new(2);
    let mut w = p.workspace(&a.view().try_finite().unwrap()).unwrap();
    let mut result = PartialSvdBuffer::new(4, 3, 2).unwrap();
    let result_report = p
        .solve_into(
            &a.view().try_finite().unwrap(),
            &[v(1., 0.), v(2., 0.5), v(1., 0.)],
            &mut result,
            &mut w,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(
        result_report.termination(),
        SpectralTermination::Converged,
        "{result:?}"
    );
    let tol = S::Real::try_from_usize_exact(2048).unwrap() * <S::Real as RealScalar>::EPSILON;
    assert!((result.singular_values()[0] - S::Real::try_from_usize_exact(3).unwrap()).abs() < tol);
    let sketch = Matrix::from_vec(
        3,
        2,
        vec![
            v(1., 0.),
            v(1., 0.),
            v(1., 0.),
            v(-1., 0.),
            v(1., 0.),
            v(0., 0.),
        ],
    )
    .unwrap();
    let probes = Matrix::from_vec(3, 1, vec![v(1., 0.), v(3., 0.), v(2., 0.)]).unwrap();
    let p = RandomizedLowRank::new(2).with_power_steps(1);
    let mut w = p.workspace(&a.view().try_finite().unwrap(), 2).unwrap();
    let mut out = RandomizedSvdBuffer::new(4, 3, 2).unwrap();
    let out_report = p
        .svd_into(
            &a.view().try_finite().unwrap(),
            sketch.view(),
            probes.view(),
            &mut out,
            &mut w,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(out_report.termination(), SpectralTermination::Converged);
    let mut out = InterpolativeBuffer::new(4, 3, 2).unwrap();
    let out_report = p
        .interpolative_into(
            &a.view().try_finite().unwrap(),
            sketch.view(),
            probes.view(),
            &mut out,
            &mut w,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(out_report.termination(), SpectralTermination::Converged);
    let mut a = Matrix::<S>::zeros(4, 4).unwrap();
    for i in 0..4 {
        a.view_mut().set(i, i, v((i + 1) as f64, 0.)).unwrap();
    }
    let p = LobpcgSolver::new(1);
    let mut w = p.workspace(&a.view().try_finite().unwrap(), 0).unwrap();
    let mut out = LobpcgBuffer::new(4, 1).unwrap();
    let x = [v(1., 0.), v(2., 1.), v(3., -1.), v(4., 0.)];
    let out_report = p
        .solve_into(
            &a.view().try_finite().unwrap(),
            BlockVectorView::new(&x, 4, 1).unwrap(),
            BlockVectorView::new(&[], 4, 0).unwrap(),
            &mut out,
            &mut w,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(
        out_report.termination(),
        SpectralTermination::Converged,
        "{out:?}"
    );
    assert!((out.eigenvalues()[0] - S::Real::ONE).abs() < tol);
}
#[test]
fn spectral_real32() {
    all_spectral_domains::<f32>()
}
#[test]
fn spectral_real64() {
    all_spectral_domains::<f64>()
}
#[test]
fn spectral_complex32() {
    all_spectral_domains::<Complex<f32>>()
}
#[test]
fn spectral_complex64() {
    all_spectral_domains::<Complex<f64>>()
}
#[test]
fn action_plans_run_in_root_and_child_epochs() {
    use scioxide_core::{
        AttemptOutcome, BudgetKind, EvaluationRole, ProblemToken, SolveBudgets, SolveIdentity,
        SolveState,
    };
    use scioxide_linalg::{PartialSvdBuffer, PartialSvdSolver};
    let a = Matrix::from_vec(3, 3, vec![3., 0., 0., 0., 2., 0., 0., 0., 1.]).unwrap();
    let a = a.view().try_finite().unwrap();
    let plan = PartialSvdSolver::new(1);
    let mut w = plan.workspace(&a).unwrap();
    let mut out = PartialSvdBuffer::new(3, 3, 1).unwrap();
    let out_report = plan
        .solve_into(
            &a,
            &[1., 1., 1.],
            &mut out,
            &mut w,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(out_report.termination(), SpectralTermination::Converged);
    let root = w.last_attempt().unwrap();
    assert_eq!(root.outcome(), AttemptOutcome::Succeeded);
    let root_epoch = root.epoch().unwrap();
    // A child of a parent with two shared evaluations stops with verified progress.
    let mut state = SolveState::new();
    let mut budgets =
        SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, Budget::try_from(2).unwrap());
    let mut parent = state
        .begin_solve(
            ExecutionContext::sequential(),
            SolveIdentity::new(
                ProblemToken::new(1),
                scioxide_core::algorithm_id!("tests", "action_parent", 1),
                0,
                0,
            ),
            &mut budgets,
        )
        .unwrap();
    let out_report = plan
        .solve_into(&a, &[1., 1., 1.], &mut out, &mut w, &mut parent)
        .unwrap();
    assert_eq!(
        out_report.termination(),
        SpectralTermination::EvaluationLimit
    );
    let child = w.last_attempt().unwrap();
    assert_eq!(child.outcome(), AttemptOutcome::Stopped);
    assert_ne!(child.epoch().unwrap(), root_epoch);
    assert_eq!(
        parent
            .counts()
            .nested()
            .provider(EvaluationRole::LinearOperator),
        out_report.applications()
    );
    assert!(out_report.applications() <= 2);
    // A shape error is a preflight rejection that starts no epoch and keeps the output.
    let before = out_report.termination();
    assert_eq!(
        plan.solve_into(
            &a,
            &[1., 1.],
            &mut out,
            &mut w,
            ExecutionContext::sequential()
        )
        .unwrap_err()
        .kind(),
        scioxide_core::SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(
        w.last_attempt().unwrap().outcome(),
        AttemptOutcome::PreflightRejected
    );
    assert_eq!(out_report.termination(), before);
}

// A = Q diag(d) Q with Q = I - 2 11ᵀ/n. Its action has a closed form,
// independent of SciOxide's dense eigenproblem and matrix-function implementations.
fn reflected_diagonal(d: &[f64]) -> Matrix<f64> {
    let n = d.len();
    let mean = d.iter().sum::<f64>() / n as f64;
    Matrix::from_vec(
        n,
        n,
        (0..n)
            .flat_map(|i| {
                (0..n).map(move |j| {
                    (if i == j { d[i] } else { 0. }) - 2. * (d[i] + d[j] - 2. * mean) / n as f64
                })
            })
            .collect(),
    )
    .unwrap()
}
fn reflected_action(d: &[f64], b: &[f64], f: impl Fn(f64) -> f64) -> Vec<f64> {
    let mean = b.iter().sum::<f64>() / b.len() as f64;
    let y: Vec<_> = d
        .iter()
        .zip(b)
        .map(|(&d, &b)| f(d) * (b - 2. * mean))
        .collect();
    let mean = y.iter().sum::<f64>() / y.len() as f64;
    y.into_iter().map(|v| v - 2. * mean).collect()
}
fn difference_norm(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt()
}

#[test]
fn default_exponential_is_rhs_scale_invariant_against_closed_form() {
    let n = 30;
    let d: Vec<_> = (0..n).map(|i| -0.1 - i as f64 / n as f64).collect();
    let a = reflected_diagonal(&d);
    let op = a.view().try_finite().unwrap();
    let plan = ExponentialAction::new(4.);
    let mut w = plan.workspace(&op, 1).unwrap();
    let b: Vec<_> = (0..n).map(|i| if i % 2 == 0 { 1. } else { -1. }).collect();
    let reference = reflected_action(&d, &b, f64::exp);
    let reference_norm = reference.iter().map(|x| x * x).sum::<f64>().sqrt();
    let mut baseline: Option<Vec<f64>> = None;
    for scale in [1., 1e-10, 1e10] {
        let rhs: Vec<_> = b.iter().map(|b| b * scale).collect();
        let mut output = vec![0.; n];
        let report = plan
            .apply_into(
                &op,
                scioxide_linalg::PhiFunction::phi(0, 1.),
                BlockVectorView::new(&rhs, n, 1).unwrap(),
                BlockVectorViewMut::new(&mut output, n, 1).unwrap(),
                &mut w,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert_eq!(report.termination(), SpectralTermination::Converged);
        for value in &mut output {
            *value /= scale;
        }
        assert!(difference_norm(&output, &reference) / reference_norm <= 1e-14);
        if let Some(previous) = &baseline {
            assert!(difference_norm(&output, previous) / reference_norm <= 1e-14);
        } else {
            baseline = Some(output);
        }
    }
}

#[test]
fn function_action_converges_with_one_or_two_cycle_capacity() {
    use scioxide_linalg::{FunctionAction, MatrixFunction};
    let n = 80;
    let d: Vec<_> = (0..n).map(|i| 0.5 + 2.5 * i as f64 / n as f64).collect();
    let a = reflected_diagonal(&d);
    let op = a.view().try_finite().unwrap();
    let b: Vec<_> = (0..n).map(|i| if i % 2 == 0 { 1. } else { -1. }).collect();
    let reference = reflected_action(&d, &b, f64::sqrt);
    for (m, p) in [(40, 40), (30, 60)] {
        let plan = FunctionAction::default()
            .with_restart(Budget::try_from(m).unwrap())
            .with_projection(Budget::try_from(p).unwrap());
        let mut w = plan.workspace(&op, 1).unwrap();
        for scale in [1., 1e-10, 1e10] {
            let rhs: Vec<_> = b.iter().map(|x| x * scale).collect();
            let mut output = vec![0.; n];
            let report = plan
                .apply_into(
                    &op,
                    MatrixFunction::Sqrt,
                    BlockVectorView::new(&rhs, n, 1).unwrap(),
                    BlockVectorViewMut::new(&mut output, n, 1).unwrap(),
                    &mut w,
                    ExecutionContext::sequential(),
                )
                .unwrap();
            assert_eq!(report.termination(), SpectralTermination::Converged);
            assert!(report.cycles() <= p / m);
            for value in &mut output {
                *value /= scale;
            }
            let error = difference_norm(&output, &reference);
            let indicator = w.last_accepted().unwrap().estimated_errors()[0] / scale;
            assert!(
                indicator >= error / 100. && indicator <= error * 100.,
                "m={m} error={error} indicator={indicator}"
            );
        }
    }
}

#[test]
fn real_function_actions_accept_imaginary_errors_within_the_error_indicator() {
    use scioxide_linalg::{FunctionAction, MatrixFunction};
    // cos(A) b is real for symmetric A. f(H) comes from a complex Schur form,
    // so unconverged restarts carry imaginary errors of the size of their
    // error indicator, which must not be taken for a complex result.
    let n = 60;
    let d: Vec<_> = (0..n)
        .map(|i| 1. + 99. * (i as f64 / (n - 1) as f64).powi(2))
        .collect();
    let a = reflected_diagonal(&d);
    let op = a.view().try_finite().unwrap();
    let b: Vec<_> = (0..n).map(|i| if i % 2 == 0 { 1. } else { -1. }).collect();
    let plan = FunctionAction::default()
        .with_restart(Budget::try_from(6).unwrap())
        .with_projection(Budget::try_from(n).unwrap());
    let block = BlockVectorView::new(&b, n, 1).unwrap();
    let (complex, complex_report) = plan
        .apply::<f64, Complex<f64>>(&op, MatrixFunction::Cos, block)
        .unwrap();
    let (real, real_report) = plan
        .apply::<f64, f64>(&op, MatrixFunction::Cos, block)
        .unwrap();
    assert_eq!(real_report.termination(), complex_report.termination());
    for (&x, z) in real.iter().zip(&complex) {
        assert_eq!(x, z.re);
    }
}

#[test]
fn hermitian_structure_check_allows_epsilon_level_residual_tolerance() {
    use scioxide_core::ResidualTolerance;
    use scioxide_linalg::{OperatorStructure, PartialSpectrumBuffer, PartialSpectrumSolver};
    let n = 120;
    let d: Vec<_> = (0..n).map(|i| if i == 0 { 20. } else { 1. }).collect();
    let mut a = Matrix::<f64>::zeros(n, n).unwrap();
    for (i, &value) in d.iter().enumerate() {
        a.view_mut().set(i, i, value).unwrap();
    }
    let op = a.view().try_finite().unwrap();
    // Unfused residual evaluation rounds `lambda * v`, so acceptance needs a few
    // epsilons; the structure check must still accept this tolerance.
    let tolerance = ResidualTolerance::new(0., 4. * f64::EPSILON).unwrap();
    let plan = PartialSpectrumSolver::new(1)
        .with_structure(OperatorStructure::Hermitian)
        .with_tolerance(tolerance);
    let mut w = plan.workspace(&op).unwrap();
    let mut out = PartialSpectrumBuffer::new(n, 1).unwrap();
    let out_report = plan
        .solve_into(
            &op,
            vec![1.; n].as_slice(),
            &mut out,
            &mut w,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(
        out_report.termination(),
        SpectralTermination::Converged,
        "residuals={:?}",
        out.residuals()
    );
    assert!((out.eigenvalues()[0].re - 20.).abs() < 1e-13);
    assert!(out.residuals()[0] <= 4. * f64::EPSILON * 20.);
}
