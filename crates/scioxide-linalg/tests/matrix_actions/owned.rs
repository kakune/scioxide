//! Owned action conveniences prepare storage and preserve borrowed-form results.
use super::*;
use scioxide_linalg::*;

fn close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!((actual - expected).abs() <= 1e-10 * expected.abs().max(1.));
    }
}

fn norm<S: NumericScalar>() {
    for n in [0, 1, 2] {
        let entries = match n {
            0 => vec![],
            1 => vec![3.],
            _ => vec![1., 2., 3., 4.],
        };
        let a = Matrix::from_vec(
            n,
            n,
            entries
                .into_iter()
                .map(|x| S::from_real(S::Real::try_from_f64_exact(x).unwrap()))
                .collect(),
        )
        .unwrap();
        let op = a.view().try_finite().unwrap();
        let plan = OneNormEstimator::new();
        let owned = plan.estimate(&op).unwrap();
        let mut workspace = plan.workspace(&op).unwrap();
        let scoped = plan
            .estimate_scoped(&op, &mut workspace, ExecutionContext::sequential_native())
            .unwrap();
        let tolerance = S::Real::EPSILON * S::Real::try_from_usize_exact(32).unwrap();
        assert!((owned.estimate() - scoped.estimate()).abs() <= tolerance);
        let expected = S::Real::try_from_f64_exact([0., 3., 6.][n]).unwrap();
        assert!((owned.estimate() - expected).abs() <= tolerance);
        assert_eq!(owned.termination(), scoped.termination());
        assert_eq!(owned.applications(), scoped.applications());
    }
}

#[test]
fn owned_one_norm_handles_zero_singleton_real_and_complex_operators() {
    norm::<f32>();
    norm::<f64>();
    norm::<Complex<f32>>();
    norm::<Complex<f64>>();
    let a = Matrix::from_vec(
        2,
        2,
        vec![
            Complex::new(3., 4.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(1., 0.),
        ],
    )
    .unwrap();
    let op = a.view().try_finite().unwrap();
    let plan = OneNormEstimator::new();
    let owned = plan.estimate(&op).unwrap();
    let scoped = plan
        .estimate_scoped(
            &op,
            &mut plan.workspace(&op).unwrap(),
            ExecutionContext::sequential_native(),
        )
        .unwrap();
    assert!((owned.estimate() - 5_f64).abs() < 1e-12);
    assert!((owned.estimate() - scoped.estimate()).abs() < 1e-12);
}

#[test]
fn owned_exponential_and_phi_preserve_column_major_blocks_and_limit_reports() {
    let a = Matrix::from_vec(2, 2, vec![0., 1., 0., 0.]).unwrap();
    let op = a.view().try_finite().unwrap();
    let input = BlockVectorView::new(&[1., 2., 3., 4.], 2, 2).unwrap();
    for limit in [false, true] {
        let options = ExponentialAction::new(1.).with_max_phi(1);
        let options = if limit {
            options.with_applications(Budget::ONE)
        } else {
            options
        };
        let plan = options;
        for function in [PhiFunction::exp(1.), PhiFunction::phi(1, 1.)] {
            let (owned, report) = plan.apply(&op, function, input).unwrap();
            let mut borrowed = [0.; 4];
            let scoped = plan
                .apply_into(
                    &op,
                    function,
                    input,
                    BlockVectorViewMut::new(&mut borrowed, 2, 2).unwrap(),
                    &mut plan.workspace(&op, 2).unwrap(),
                    ExecutionContext::sequential_native(),
                )
                .unwrap();
            close(&owned, &borrowed);
            assert_eq!(report.termination(), scoped.termination());
            assert_eq!(report.applications(), scoped.applications());
            if limit {
                assert_eq!(report.termination(), SpectralTermination::EvaluationLimit);
            } else {
                assert_eq!(report.termination(), SpectralTermination::Converged);
                close(
                    &owned,
                    if function.index() == 0 {
                        &[3., 2., 7., 4.]
                    } else {
                        &[2., 2., 5., 4.]
                    },
                );
            }
        }
    }
}

#[test]
fn owned_function_action_preserves_output_scalar_blocks_and_limits() {
    let a = Matrix::from_vec(2, 2, vec![4., 0., 0., 9.]).unwrap();
    let op = a.view().try_finite().unwrap();
    let input = BlockVectorView::new(&[1., 2., 3., 4.], 2, 2).unwrap();
    for limit in [false, true] {
        let options = FunctionAction::default();
        let options = if limit {
            options.with_applications(Budget::ONE)
        } else {
            options
        };
        let plan = options;
        let (owned, report) = plan
            .apply::<f64, f64>(&op, MatrixFunction::Sqrt, input)
            .unwrap();
        let mut borrowed = [0.; 4];
        let scoped = plan
            .apply_into(
                &op,
                MatrixFunction::Sqrt,
                input,
                BlockVectorViewMut::new(&mut borrowed, 2, 2).unwrap(),
                &mut plan.workspace(&op, 2).unwrap(),
                ExecutionContext::sequential_native(),
            )
            .unwrap();
        close(&owned, &borrowed);
        assert_eq!(report.termination(), scoped.termination());
        if limit {
            assert_eq!(report.termination(), SpectralTermination::EvaluationLimit);
        } else {
            assert_eq!(report.termination(), SpectralTermination::Converged);
            close(&owned, &[2., 6., 6., 12.]);
            let (complex, _) = plan
                .apply::<f64, Complex<f64>>(&op, MatrixFunction::Sqrt, input)
                .unwrap();
            for (actual, expected) in complex.iter().zip(&owned) {
                assert!((actual.re - expected).abs() < 1e-10);
                assert!(actual.im.abs() < 1e-10);
            }
        }
    }
}

#[test]
fn owned_spectrum_svd_and_lobpcg_match_reserved_forms() {
    let a = Matrix::from_vec(2, 2, vec![4., 0., 0., 1.]).unwrap();
    let op = a.view().try_finite().unwrap();
    for limit in [false, true] {
        let spectrum = PartialSpectrumSolver::new(1).with_structure(OperatorStructure::Hermitian);
        let spectrum = if limit {
            spectrum.with_applications(Budget::ONE)
        } else {
            spectrum
        };
        let plan = spectrum;
        let (owned, report) = plan.solve(&op, &[1., 1.][..]).unwrap();
        let mut borrowed = PartialSpectrumBuffer::new(2, 1).unwrap();
        let scoped = plan
            .solve_into(
                &op,
                &[1., 1.][..],
                &mut borrowed,
                &mut plan.workspace(&op).unwrap(),
                ExecutionContext::sequential_native(),
            )
            .unwrap();
        assert_eq!(report.termination(), scoped.termination());
        assert_eq!(owned.eigenvalues().len(), borrowed.eigenvalues().len());
        for (actual, expected) in owned.eigenvalues().iter().zip(borrowed.eigenvalues()) {
            assert!((*actual - *expected).abs() < 1e-10);
        }
        close(owned.residuals(), borrowed.residuals());

        let svd = PartialSvdSolver::new(1);
        let svd = if limit {
            svd.with_applications(Budget::ONE)
        } else {
            svd
        };
        let plan = svd;
        let (owned, report) = plan.solve(&op, &[1., 1.]).unwrap();
        let mut borrowed = PartialSvdBuffer::new(2, 2, 1).unwrap();
        let scoped = plan
            .solve_into(
                &op,
                &[1., 1.],
                &mut borrowed,
                &mut plan.workspace(&op).unwrap(),
                ExecutionContext::sequential_native(),
            )
            .unwrap();
        assert_eq!(report.termination(), scoped.termination());
        close(owned.singular_values(), borrowed.singular_values());
        if !limit {
            close(owned.singular_values(), &[4.]);
        }

        let lobpcg = LobpcgSolver::new(1).with_selection(LobpcgSelection::Largest);
        let lobpcg = if limit {
            lobpcg.with_applications(Budget::ONE)
        } else {
            lobpcg
        };
        let plan = lobpcg;
        let initial = BlockVectorView::new(&[1., 1.], 2, 1).unwrap();
        let constraints = BlockVectorView::new(&[], 2, 0).unwrap();
        let (owned, report) = plan.solve(&op, initial, constraints).unwrap();
        let mut borrowed = LobpcgBuffer::new(2, 1).unwrap();
        let scoped = plan
            .solve_into(
                &op,
                initial,
                constraints,
                &mut borrowed,
                &mut plan.workspace(&op, 0).unwrap(),
                ExecutionContext::sequential_native(),
            )
            .unwrap();
        assert_eq!(report.termination(), scoped.termination());
        close(owned.eigenvalues(), borrowed.eigenvalues());
        if !limit {
            close(owned.eigenvalues(), &[4.]);
        }
        close(owned.residuals(), borrowed.residuals());
    }
}

#[test]
fn owned_low_rank_forms_match_reserved_forms_and_expose_only_completed_factors() {
    let a = Matrix::from_vec(2, 2, vec![2., 0., 0., 0.]).unwrap();
    let op = a.view().try_finite().unwrap();
    let sketch = Matrix::from_vec(2, 1, vec![1., 1.]).unwrap();
    let probes = Matrix::from_vec(2, 1, vec![1., 0.]).unwrap();
    for limit in [false, true] {
        let options = RandomizedLowRank::new(1);
        let options = if limit {
            options.with_applications(Budget::ONE)
        } else {
            options
        };
        let plan = options;
        let (owned, report) = plan.svd(&op, sketch.view(), probes.view()).unwrap();
        let mut borrowed = RandomizedSvdBuffer::new(2, 2, 1).unwrap();
        let scoped = plan
            .svd_into(
                &op,
                sketch.view(),
                probes.view(),
                &mut borrowed,
                &mut plan.workspace(&op, 1).unwrap(),
                ExecutionContext::sequential_native(),
            )
            .unwrap();
        assert_eq!(report.termination(), scoped.termination());
        assert_eq!(report.rank(), scoped.rank());
        close(owned.singular_values(), borrowed.singular_values());
        if !limit {
            close(owned.singular_values(), &[2.]);
        }
        let (owned, report) = plan
            .interpolative(&op, sketch.view(), probes.view())
            .unwrap();
        let mut borrowed = InterpolativeBuffer::new(2, 2, 1).unwrap();
        let scoped = plan
            .interpolative_into(
                &op,
                sketch.view(),
                probes.view(),
                &mut borrowed,
                &mut plan.workspace(&op, 1).unwrap(),
                ExecutionContext::sequential_native(),
            )
            .unwrap();
        assert_eq!(report.termination(), scoped.termination());
        assert_eq!(report.rank(), scoped.rank());
        assert_eq!(owned.column_indices(), borrowed.column_indices());
        if limit {
            assert_eq!(report.termination(), SpectralTermination::EvaluationLimit);
            assert_eq!(owned.columns().cols(), 0);
            assert_eq!(owned.coefficients().rows(), 0);
        } else {
            assert_eq!(report.termination(), SpectralTermination::Converged);
            assert_eq!(owned.column_indices(), [0]);
            for i in 0..2 {
                for j in 0..2 {
                    let reconstructed = owned.columns().get(i, 0).unwrap()
                        * owned.coefficients().get(0, j).unwrap();
                    assert!((reconstructed - a.view().get(i, j).unwrap()).abs() < 1e-10);
                }
            }
        }
    }
}
