//! Empty jobs and finite spectra/factors at both ends of scalar range.
use super::*;
use scioxide_linalg::*;

fn extremes<S: NumericScalar>(scales: &[f64])
where
    Complex<S::Real>: NumericScalar<Real = S::Real>,
{
    let c = |x| S::Real::try_from_f64_lossy(x).unwrap();
    let v = |x| S::from_real(c(x));
    let tolerance = scioxide_core::ResidualTolerance::new(
        S::Real::ZERO,
        c(512.) * <S::Real as RealScalar>::EPSILON,
    )
    .unwrap();
    for &scale in scales {
        let a = Matrix::from_vec(
            3,
            3,
            vec![
                v(scale),
                S::ZERO,
                S::ZERO,
                S::ZERO,
                v(2. * scale),
                S::ZERO,
                S::ZERO,
                S::ZERO,
                v(3. * scale),
            ],
        )
        .unwrap();
        let a = a.view().try_finite().unwrap();
        let p = PartialSpectrumSolver::new(1).with_tolerance(tolerance);
        let mut w = p.workspace(&a).unwrap();
        let mut out = PartialSpectrumBuffer::new(3, 1).unwrap();
        let out_report = p
            .solve_into(
                &a,
                [S::ONE; 3].as_slice(),
                &mut out,
                &mut w,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert_eq!(out_report.termination(), SpectralTermination::Converged);
        assert!(
            (out.eigenvalues()[0].re / c(scale) - c(3.)).abs()
                < c(4096.) * <S::Real as RealScalar>::EPSILON
        );
        let p = PartialSvdSolver::new(1).with_tolerance(tolerance);
        let mut w = p.workspace(&a).unwrap();
        let mut out = PartialSvdBuffer::new(3, 3, 1).unwrap();
        let out_report = p
            .solve_into(
                &a,
                &[S::ONE; 3],
                &mut out,
                &mut w,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert_eq!(out_report.termination(), SpectralTermination::Converged);
        assert!(
            (out.singular_values()[0] / c(scale) - c(3.)).abs()
                < c(4096.) * <S::Real as RealScalar>::EPSILON
        );
        let p = LobpcgSolver::new(1).with_tolerance(tolerance);
        let mut w = p.workspace(&a, 0).unwrap();
        let mut out = LobpcgBuffer::new(3, 1).unwrap();
        let out_report = p
            .solve_into(
                &a,
                BlockVectorView::new(&[S::ONE; 3], 3, 1).unwrap(),
                BlockVectorView::new(&[], 3, 0).unwrap(),
                &mut out,
                &mut w,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert_eq!(
            out_report.termination(),
            SpectralTermination::Converged,
            "scale {scale}: {out:?}"
        );
        assert!(
            (out.eigenvalues()[0] / c(scale) - S::Real::ONE).abs()
                < c(4096.) * <S::Real as RealScalar>::EPSILON
        );
        let a = Matrix::from_vec(
            3,
            3,
            vec![
                v(3. * scale),
                S::ZERO,
                S::ZERO,
                S::ZERO,
                v(2. * scale),
                S::ZERO,
                S::ZERO,
                S::ZERO,
                S::ZERO,
            ],
        )
        .unwrap();
        let a = a.view().try_finite().unwrap();
        let sketch = Matrix::from_vec(
            3,
            2,
            vec![S::ONE, S::ZERO, S::ZERO, S::ONE, S::ZERO, S::ZERO],
        )
        .unwrap();
        let probes = Matrix::identity(3).unwrap();
        let p = RandomizedLowRank::new(2).with_tolerance(tolerance);
        let mut w = p.workspace(&a, 2).unwrap();
        let mut out = RandomizedSvdBuffer::new(3, 3, 2).unwrap();
        let out_report = p
            .svd_into(
                &a,
                sketch.view(),
                probes.view(),
                &mut out,
                &mut w,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert_eq!(out_report.termination(), SpectralTermination::Converged);
        for (&sigma, expected) in out.singular_values().iter().zip([3., 2.]) {
            assert!(
                (sigma / c(scale) - c(expected)).abs()
                    < c(4096.) * <S::Real as RealScalar>::EPSILON
            );
        }
        let mut id = InterpolativeBuffer::new(3, 3, 2).unwrap();
        let id_report = p
            .interpolative_into(
                &a,
                sketch.view(),
                probes.view(),
                &mut id,
                &mut w,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert_eq!(id_report.termination(), SpectralTermination::Converged);
    }
}
#[test]
fn extreme_real64() {
    extremes::<f64>(&[1e-300, 1e300]);
}
#[test]
fn extreme_real32() {
    extremes::<f32>(&[1e-30, 1e30]);
}
#[test]
fn extreme_complex64() {
    extremes::<Complex<f64>>(&[1e-300, 1e300]);
}
#[test]
fn extreme_complex32() {
    extremes::<Complex<f32>>(&[1e-30, 1e30]);
}

#[test]
fn empty_operator_jobs_publish_empty_identities() {
    use core::cell::Cell;
    use scioxide_core::{AdjointOperator, LinearOperator, SciOxideError};
    struct Empty(Cell<usize>);
    impl LinearOperator<f64> for Empty {
        fn rows(&self) -> usize {
            0
        }
        fn cols(&self) -> usize {
            0
        }
        fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
            Ok(0)
        }
        fn apply_into(&self, _: &[f64], _: &mut [f64], _: &mut [f64]) -> Result<(), SciOxideError> {
            self.0.set(self.0.get() + 1);
            Ok(())
        }
    }
    impl AdjointOperator<f64> for Empty {
        fn apply_adjoint_workspace_len(&self) -> Result<usize, SciOxideError> {
            Ok(0)
        }
        fn apply_adjoint_into(
            &self,
            x: &[f64],
            y: &mut [f64],
            w: &mut [f64],
        ) -> Result<(), SciOxideError> {
            self.apply_into(x, y, w)
        }
    }
    let a = Empty(Cell::new(0));
    let execution = LinalgExecutionOptions::new().with_iteration_budget(Budget::ONE);
    let p = PartialSpectrumSolver::new(0).with_execution_options(execution);
    let mut out = PartialSpectrumBuffer::new(0, 0).unwrap();
    let report = p
        .solve_into(
            &a,
            [].as_slice(),
            &mut out,
            &mut p.workspace(&a).unwrap(),
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(report.termination(), SpectralTermination::Converged);
    assert_eq!(report.cycles(), 0);
    assert!(out.eigenvalues().is_empty());
    let p = PartialSvdSolver::new(0).with_execution_options(execution);
    let mut out = PartialSvdBuffer::new(0, 0, 0).unwrap();
    let report = p
        .solve_into(
            &a,
            &[],
            &mut out,
            &mut p.workspace(&a).unwrap(),
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(report.termination(), SpectralTermination::Converged);
    assert_eq!(report.cycles(), 0);
    assert!(out.singular_values().is_empty());
    let p = LobpcgSolver::new(0).with_execution_options(execution);
    let mut out = LobpcgBuffer::new(0, 0).unwrap();
    let report = p
        .solve_into(
            &a,
            BlockVectorView::new(&[], 0, 0).unwrap(),
            BlockVectorView::new(&[], 0, 0).unwrap(),
            &mut out,
            &mut p.workspace(&a, 0).unwrap(),
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(report.termination(), SpectralTermination::Converged);
    assert_eq!(report.iterations(), 0);
    assert!(out.eigenvalues().is_empty());
    let p = RandomizedLowRank::new(0);
    let matrix = Matrix::zeros(0, 0).unwrap();
    let mut out = RandomizedSvdBuffer::new(0, 0, 0).unwrap();
    let out_report = p
        .svd_into(
            &a,
            matrix.view(),
            Matrix::zeros(0, 1).unwrap().view(),
            &mut out,
            &mut p.workspace(&a, 0).unwrap(),
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(out_report.rank(), 0);
    assert_eq!(a.0.get(), 0);
}

#[test]
fn exponential_accepts_finite_rhs_with_overflowing_one_norm() {
    let a = Matrix::from_vec(2, 2, vec![-1., 0., 0., -1.]).unwrap();
    let op = a.view().try_finite().unwrap();
    let plan = ExponentialAction::new(1.);
    let mut w = plan.workspace(&op, 1).unwrap();
    let mut output = [0.; 2];
    let report = plan
        .apply_into(
            &op,
            scioxide_linalg::PhiFunction::phi(0, 1.),
            BlockVectorView::new(&[1e308; 2], 2, 1).unwrap(),
            BlockVectorViewMut::new(&mut output, 2, 1).unwrap(),
            &mut w,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(report.termination(), SpectralTermination::Converged);
    assert!(
        output
            .iter()
            .all(|x| x.is_finite() && (x / 1e308 - (-1f64).exp()).abs() < 1e-15)
    );
}

#[test]
fn complex_function_results_need_complex_output() {
    // sqrt(-1) and sqrt(-4) are imaginary: real output is outside the domain.
    let a = Matrix::from_vec(2, 2, vec![-1., 0., 0., -4.]).unwrap();
    let op = a.view().try_finite().unwrap();
    let input = [1., 1.];
    let plan = FunctionAction::default();
    let block = BlockVectorView::new(&input, 2, 1).unwrap();
    let error = plan
        .apply::<f64, f64>(&op, MatrixFunction::Sqrt, block)
        .unwrap_err();
    assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::DomainError);
    let (complex, _) = plan
        .apply::<f64, Complex<f64>>(&op, MatrixFunction::Sqrt, block)
        .unwrap();
    assert!((complex[0] - Complex::new(0., 1.)).norm() < 1e-12);
    assert!((complex[1] - Complex::new(0., 2.)).norm() < 1e-12);
}
