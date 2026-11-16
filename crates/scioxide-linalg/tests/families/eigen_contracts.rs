#![allow(clippy::unwrap_used)]
use scioxide_core::AlgebraicScalar;

use scioxide_core::Triangle;
use scioxide_core::{
    Complex, NumericScalar, RealScalar, ResidualTolerance, SciOxideErrorKind, Tolerance,
};
use scioxide_linalg::{
    EigenIndexRange, EigenvalueAccuracy, GeneralEigenDecomposition, GeneralEigenOptions,
    GeneralizedHermitianEigenDecomposition, HermitianEigenDecomposition, HermitianEigenOptions,
    HermitianInput, HermitianProjectionPolicy, Matrix,
};

fn rounded_gram<S: NumericScalar>() -> Matrix<S> {
    let n = 50;
    let real = |x| S::Real::try_from_f64_lossy(x).unwrap();
    let mut basis = Matrix::zeros(n, n).unwrap();
    for row in 0..n {
        let phase = if S::IS_REAL {
            0.0
        } else {
            row as f64 / n as f64
        };
        for col in 0..n {
            let value = if col == 0 {
                (1.0 / n as f64).sqrt()
            } else {
                (2.0 / n as f64).sqrt()
                    * (core::f64::consts::PI * (row as f64 + 0.5) * col as f64 / n as f64).cos()
            };
            basis.as_mut_slice()[row * n + col] =
                S::from_components(real(value * phase.cos()), real(value * phase.sin()));
        }
    }
    let mut weighted = basis.clone();
    for row in weighted.as_mut_slice().chunks_exact_mut(n) {
        for (col, value) in row.iter_mut().enumerate() {
            *value = *value * S::from_real(real(1.0 + col as f64 / n as f64));
        }
    }
    let mut gram = weighted.view().mul_mat(basis.view().adjoint()).unwrap();
    // Triangle semantics require an exactly real diagonal even when products
    // leave a tiny imaginary rounding component there.
    for i in 0..n {
        gram.as_mut_slice()[i * n + i] = S::from_real(gram.as_slice()[i * n + i].real_component());
    }
    gram
}

fn gram_contract<S: NumericScalar>() {
    let a = rounded_gram::<S>();
    let n = a.rows();
    let identity = Matrix::<S>::identity(n).unwrap();
    let bound = S::Real::try_from_usize_exact(64 * n).unwrap() * S::Real::EPSILON;
    assert!(
        (0..n).any(|i| (0..n).any(|j| a.as_slice()[i * n + j] != a.as_slice()[j * n + i].conj()))
    );
    for input in [
        HermitianInput::Triangle(Triangle::Lower),
        HermitianInput::Triangle(Triangle::Upper),
        HermitianInput::Projected(HermitianProjectionPolicy::Always),
        HermitianInput::Projected(HermitianProjectionPolicy::Within(
            Tolerance::new(S::Real::ZERO, bound).unwrap(),
        )),
    ] {
        let standard = HermitianEigenDecomposition::decompose(&a, input).unwrap();
        let standard_values =
            scioxide_linalg::HermitianEigenvaluesResult::compute(&a, input).unwrap();
        let generalized = GeneralizedHermitianEigenDecomposition::decompose(
            &a,
            &identity,
            input,
            HermitianInput::Exact,
        )
        .unwrap();
        let generalized_values = scioxide_linalg::GeneralizedHermitianEigenvaluesResult::compute(
            &a,
            &identity,
            input,
            HermitianInput::Exact,
        )
        .unwrap();
        // Exercise B's independent policy with an exactly Hermitian A.
        let inverse_values = scioxide_linalg::GeneralizedHermitianEigenvaluesResult::compute(
            &identity,
            &a,
            HermitianInput::Exact,
            input,
        )
        .unwrap();
        for (col, &value) in standard.eigenvalues().iter().enumerate() {
            let expected = S::Real::try_from_f64_lossy(1.0 + col as f64 / n as f64).unwrap();
            for actual in [
                value,
                standard_values.eigenvalues()[col],
                generalized.eigenvalues()[col],
                generalized_values.eigenvalues()[col],
            ] {
                assert!((actual - expected).abs() <= bound);
            }
            assert!(
                (inverse_values.eigenvalues()[n - 1 - col] - S::Real::ONE / expected).abs()
                    <= bound
            );
            assert!(generalized.residuals()[col] <= bound);
            for row in 0..n {
                let residual = (0..n).fold(S::ZERO, |sum, k| {
                    sum + a.as_slice()[row * n + k]
                        * standard.eigenvectors().as_slice()[k * n + col]
                }) - S::from_real(value)
                    * standard.eigenvectors().as_slice()[row * n + col];
                assert!(residual.abs() <= bound);
                let inner = (0..n).fold(S::ZERO, |sum, k| {
                    sum + standard.eigenvectors().as_slice()[k * n + row].conj()
                        * standard.eigenvectors().as_slice()[k * n + col]
                });
                assert!((inner - if row == col { S::ONE } else { S::ZERO }).abs() <= bound);
            }
        }
        let projected = matches!(input, HermitianInput::Projected(_));
        assert_eq!(standard.projection_report().is_some(), projected);
        assert_eq!(standard_values.projection_report().is_some(), projected);
        assert_eq!(generalized.a_projection_report().is_some(), projected);
        assert!(generalized.b_projection_report().is_none());
        assert_eq!(inverse_values.b_projection_report().is_some(), projected);
        if let Some(report) = standard.projection_report() {
            assert!(report.absolute_frobenius() > S::Real::ZERO);
        }
    }
    let standard = scioxide_linalg::HermitianEigenvaluesResult::compute(&a, HermitianInput::Exact)
        .unwrap_err();
    let generalized = scioxide_linalg::GeneralizedHermitianEigenvaluesResult::compute(
        &a,
        &identity,
        HermitianInput::Exact,
        HermitianInput::Exact,
    )
    .unwrap_err();
    assert_eq!(standard.kind(), SciOxideErrorKind::InvalidInput);
    assert_eq!(invalid_contract(&standard), invalid_contract(&generalized));
}

#[test]
fn rounded_products_use_explicit_input_policies_without_looser_deflation() {
    gram_contract::<f64>();
    gram_contract::<f32>();
    gram_contract::<Complex<f64>>();
    gram_contract::<Complex<f32>>();
}

#[test]
fn residual_acceptance_is_independent_of_deflation() {
    let a = Matrix::from_vec(2, 2, vec![1.0_f64, 1.0, 1e-6, 2.0]).unwrap();
    let defaults = GeneralEigenOptions::new();
    let _ = GeneralEigenDecomposition::decompose(&a).unwrap();
    let deflation = defaults.with_deflation_tolerance(Tolerance::new(1e-3, 0.0).unwrap());
    assert_eq!(
        deflation.residual_tolerance(),
        defaults.residual_tolerance()
    );
    assert_eq!(
        GeneralEigenDecomposition::decompose_with_options(&a, &deflation)
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::NumericalInstability
    );
    let acceptance = deflation.with_residual_tolerance(ResidualTolerance::new(1e-4, 0.0).unwrap());
    assert_eq!(
        acceptance.deflation_tolerance(),
        deflation.deflation_tolerance()
    );
    let _ = GeneralEigenDecomposition::decompose_with_options(&a, &acceptance).unwrap();
    assert_eq!(
        scioxide_linalg::GeneralEigenvaluesResult::compute_with_options(&a, &acceptance)
            .unwrap()
            .eigenvalues(),
        scioxide_linalg::GeneralEigenvaluesResult::compute_with_options(&a, &deflation)
            .unwrap()
            .eigenvalues()
    );
}

fn invalid_contract(
    error: &scioxide_core::SciOxideError,
) -> (
    SciOxideErrorKind,
    Option<(scioxide_core::Argument, scioxide_core::InvalidInputReason)>,
) {
    let detail = match error.view() {
        scioxide_core::SciOxideErrorView::InvalidInput {
            argument, reason, ..
        } => Some((argument, reason)),
        _ => None,
    };
    (error.kind(), detail)
}

#[test]
fn a_complex_diagonal_requires_explicit_projection() {
    let a = Matrix::from_vec(1, 1, vec![Complex::new(2.0_f64, 1e-14)]).unwrap();
    let b = Matrix::<Complex<f64>>::identity(1).unwrap();
    for input in [
        HermitianInput::Exact,
        HermitianInput::Triangle(Triangle::Lower),
    ] {
        let standard = scioxide_linalg::HermitianEigenvaluesResult::compute(&a, input).unwrap_err();
        let generalized = scioxide_linalg::GeneralizedHermitianEigenvaluesResult::compute(
            &a,
            &b,
            input,
            HermitianInput::Exact,
        )
        .unwrap_err();
        assert_eq!(invalid_contract(&standard), invalid_contract(&generalized));
        assert_eq!(standard.kind(), SciOxideErrorKind::InvalidInput);
    }
    let projected = scioxide_linalg::HermitianEigenvaluesResult::compute(
        &a,
        HermitianInput::Projected(HermitianProjectionPolicy::Always),
    )
    .unwrap();
    assert_eq!(projected.eigenvalues(), &[2.0]);
    assert!((projected.projection_report().unwrap().absolute_frobenius() - 1e-14).abs() < 1e-28);
}

#[test]
fn deflation_options_do_not_regularize_pivots_or_reject_independent_vectors() {
    let a = Matrix::from_vec(2, 2, vec![1.0_f64, 1e-6, 0.0, 1.0 + 1e-8]).unwrap();
    let options =
        GeneralEigenOptions::new().with_deflation_tolerance(Tolerance::new(1e-3, 0.0).unwrap());
    let eigen = GeneralEigenDecomposition::decompose_with_options(&a, &options).unwrap();
    assert_eq!(eigen.eigenvalues().len(), 2);
    assert!((eigen.eigenvalues()[1].re - (1.0 + 1e-8)).abs() < 8.0 * f64::EPSILON);
}

#[test]
fn hermitian_eigenvalue_restoration_overflow_is_numerical_instability() {
    use scioxide_core::{NumericalIssue, SciOxideErrorView};
    use scioxide_linalg::{HermitianEigenvaluesResult, HermitianInput};
    let overflow = |error: scioxide_core::SciOxideError| {
        assert!(
            matches!(
                error.view(),
                SciOxideErrorView::NumericalInstability {
                    issue: NumericalIssue::Overflow,
                    ..
                }
            ),
            "{error:?}"
        );
    };
    // The eigenvalue 2 MAX of [[MAX, MAX], [MAX, MAX]] exceeds the finite range.
    let big = scioxide_linalg::Matrix::from_vec(2, 2, vec![f64::MAX; 4]).unwrap();
    overflow(HermitianEigenvaluesResult::compute(&big, HermitianInput::Exact).unwrap_err());
    overflow(
        HermitianEigenvaluesResult::tridiagonal(&[f64::MAX, f64::MAX], &[f64::MAX]).unwrap_err(),
    );
}

#[test]
fn generalized_pairs_name_an_indefinite_metric() {
    use scioxide_core::SciOxideErrorView;
    use scioxide_linalg::{GeneralizedHermitianEigenvaluesResult, Matrix};
    let a = Matrix::from_vec(2, 2, vec![2.0, 1.0, 1.0, 3.0]).unwrap();
    // Semidefinite and indefinite metrics both fail the unpivoted Cholesky.
    for b in [vec![1.0, 1.0, 1.0, 1.0], vec![1.0, 0.0, 0.0, -1.0]] {
        let b = Matrix::from_vec(2, 2, b).unwrap();
        let error = GeneralizedHermitianEigenvaluesResult::compute(
            &a,
            &b,
            HermitianInput::Exact,
            HermitianInput::Exact,
        )
        .unwrap_err();
        assert!(
            matches!(
                error.view(),
                SciOxideErrorView::DomainError { argument, .. } if argument == scioxide_core::argument!("b")
            ),
            "{error:?}"
        );
    }
}

#[test]
fn selected_jobs_on_a_zero_matrix_return_zero_and_unit_vectors() {
    use scioxide_linalg::{GeneralizedHermitianEigenDecomposition, HermitianEigenvaluesResult};
    let relative = HermitianEigenOptions::<f64>::default()
        .with_eigenvalue_accuracy(EigenvalueAccuracy::Relative);
    let range = |start, end| {
        HermitianEigenOptions::<f64>::default()
            .with_index_range(EigenIndexRange::new(start, end).unwrap())
    };
    // Each selected eigenvector of `T = 0` is the unit vector of its index.
    let check = |eigen: &HermitianEigenDecomposition<f64>, first: usize| {
        let vectors = eigen.eigenvectors();
        assert!(eigen.eigenvalues().iter().all(|&value| value == 0.0));
        for column in 0..vectors.cols() {
            for row in 0..vectors.rows() {
                let expected = f64::from(u8::from(row == first + column));
                assert_eq!(vectors.view().get(row, column).unwrap(), expected);
            }
        }
    };
    for options in [range(0, 2), relative] {
        check(
            &HermitianEigenDecomposition::tridiagonal_with_options(&[0.0, 0.0], &[0.0], &options)
                .unwrap(),
            options.index_range().map_or(0, EigenIndexRange::start),
        );
        let values =
            HermitianEigenvaluesResult::tridiagonal_with_options(&[0.0, 0.0], &[0.0], &options)
                .unwrap();
        assert_eq!(values.eigenvalues(), &[0.0, 0.0]);
    }
    let zero = Matrix::<f64>::zeros(3, 3).unwrap();
    for (options, first, count) in [(range(1, 2), 1, 1), (relative, 0, 3)] {
        let eigen = HermitianEigenDecomposition::decompose_with_options(
            &zero,
            HermitianInput::Exact,
            &options,
        )
        .unwrap();
        assert_eq!(eigen.eigenvalues().len(), count);
        check(&eigen, first);
    }
    let metric = Matrix::from_vec(2, 2, vec![2.0, 0.0, 0.0, 1.0]).unwrap();
    let pencil = GeneralizedHermitianEigenDecomposition::decompose_with_options(
        &Matrix::<f64>::zeros(2, 2).unwrap(),
        &metric,
        HermitianInput::Exact,
        HermitianInput::Exact,
        &relative,
    )
    .unwrap();
    assert_eq!(pencil.eigenvalues(), &[0.0, 0.0]);
}

#[test]
fn relative_accuracy_survives_a_large_largest_entry() {
    use scioxide_linalg::HermitianEigenvaluesResult;
    // Downward graded with `eᵢ² = c dᵢ dᵢ₊₁`: to a relative `10⁻⁶⁰`, the
    // eigenvalues are the pivots `dᵢ rᵢ`, `r₀ = 1`, `rᵢ = 1 − c / rᵢ₋₁`.
    let cases: [(Vec<f64>, Vec<f64>, f64); 2] = [
        (
            (0..6).map(|i| 10f64.powi(20 - 63 * i)).collect(),
            (0..5)
                .map(|i| 0.1 * (10f64.powi(20 - 63 * i) * 10f64.powi(20 - 63 * (i + 1))).sqrt())
                .collect(),
            1e-14,
        ),
        // The pivot guard `MIN_POSITIVE` bounds the accuracy of `10⁻³⁰⁰`.
        (
            (0..7).map(|i| 10f64.powi(300 - 100 * i)).collect(),
            (0..6).map(|i| 0.1 * 10f64.powi(250 - 100 * i)).collect(),
            1e-6,
        ),
    ];
    let options = HermitianEigenOptions::<f64>::default()
        .with_eigenvalue_accuracy(EigenvalueAccuracy::Relative);
    for (diagonal, off, tolerance) in cases {
        let mut ratio = 1.0;
        let mut expected: Vec<f64> = (0..diagonal.len())
            .map(|i| {
                if i > 0 {
                    let coupling = (off[i - 1] / diagonal[i - 1]) * (off[i - 1] / diagonal[i]);
                    ratio = 1.0 - coupling / ratio;
                }
                diagonal[i] * ratio
            })
            .collect();
        expected.sort_by(f64::total_cmp);
        let values =
            HermitianEigenvaluesResult::tridiagonal_with_options(&diagonal, &off, &options)
                .unwrap();
        let vectors =
            HermitianEigenDecomposition::tridiagonal_with_options(&diagonal, &off, &options)
                .unwrap();
        for computed in [values.eigenvalues(), vectors.eigenvalues()] {
            for (&value, &reference) in computed.iter().zip(&expected) {
                assert!(
                    ((value - reference) / reference).abs() <= tolerance,
                    "{value:e} vs {reference:e}"
                );
            }
        }
    }
}

#[test]
fn divide_and_conquer_budget_failures_report_the_whole_solve() {
    use scioxide_core::{Budget, SciOxideErrorView};
    let n = 60;
    for limit in [5, 40, 70] {
        let options = HermitianEigenOptions::<f64>::default();
        let options = options.with_execution_options(
            options
                .execution_options()
                .with_iteration_budget(Budget::try_from(limit).unwrap()),
        );
        let error = HermitianEigenDecomposition::tridiagonal_with_options(
            &vec![2.0; n],
            &vec![-1.0; n - 1],
            &options,
        )
        .unwrap_err();
        assert!(
            matches!(
                error.view(),
                SciOxideErrorView::ConvergenceFailure { diagnostics, .. }
                    if diagnostics.counts().own().iterations() == limit
            ),
            "{error:?}"
        );
    }
}

#[test]
fn selected_dense_vectors_back_transform_only_their_columns() {
    // Blocked complex reduction (order above 64) with a phase per row.
    let n = 70;
    let a = crate::support::decomposition::hermitian_matrix::<Complex<f64>>(n, 5);
    let options = HermitianEigenOptions::<f64>::default()
        .with_index_range(EigenIndexRange::new(10, 13).unwrap());
    let selected =
        HermitianEigenDecomposition::decompose_with_options(&a, HermitianInput::Exact, &options)
            .unwrap();
    let full = HermitianEigenDecomposition::decompose(&a, HermitianInput::Exact).unwrap();
    let vectors = selected.eigenvectors();
    assert_eq!((vectors.rows(), vectors.cols()), (n, 3));
    let norm = full
        .eigenvalues()
        .iter()
        .fold(0.0_f64, |m, x| m.max(x.abs()));
    for (column, (&value, &expected)) in selected
        .eigenvalues()
        .iter()
        .zip(&full.eigenvalues()[10..13])
        .enumerate()
    {
        assert!((value - expected).abs() <= 64.0 * n as f64 * f64::EPSILON * norm);
        let v = |row: usize| vectors.view().get(row, column).unwrap();
        for row in 0..n {
            let residual = (0..n).fold(Complex::new(0.0, 0.0), |sum, k| {
                sum + a.view().get(row, k).unwrap() * v(k)
            }) - v(row) * value;
            assert!(residual.norm() <= 64.0 * n as f64 * f64::EPSILON * norm);
        }
        let length = (0..n).map(|row| v(row).norm_sqr()).sum::<f64>();
        assert!((length - 1.0).abs() <= 8.0 * n as f64 * f64::EPSILON);
    }
}
