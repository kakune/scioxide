//! Shared direct-factor vocabulary, input interpretation and acceptance contracts.
#![allow(clippy::unwrap_used, clippy::panic)]

use crate::support::svd_cases;
use scioxide_core::Triangle;
use scioxide_core::{
    AlgebraicScalar, Complex, ExecutionContext, NumericScalar, RealScalar, Tolerance,
};
use scioxide_linalg::{
    CholeskyDecomposition, CholeskyPivoting, DirectSolveOptions, DirectSolveTermination,
    HermitianInput, HermitianProjectionPolicy, HouseholderDecomposition, HouseholderOrientation,
    HouseholderPivoting, LdlDecomposition, LinalgExecutionOptions, LinalgWorkspace, Matrix,
    RankPolicy, ReciprocalConditionThreshold, SvdDecomposition, SvdWorkspaceRequest,
    ThinSvdDecomposition, VectorViewMut,
};

macro_rules! hermitian_contract {
    ($name:ident, $factor:ident $(, $pivot:expr)?) => {
        #[test]
        fn $name() {
            let options = LinalgExecutionOptions::new();
            let input = Matrix::from_vec(2, 2, vec![4.0_f64, 2.0, 0.0, 4.0]).unwrap();
            assert!($factor::factor_with_options(&input, HermitianInput::Exact, $($pivot,)? &options).is_err());
            let mut factor = $factor::factor_with_options(&input,
                HermitianInput::Projected(HermitianProjectionPolicy::Always), $($pivot,)? &options).unwrap();
            let report = *factor.projection_report().unwrap();
            assert!((report.absolute_frobenius() - 2.0_f64.sqrt()).abs() < 1e-14);
            assert!((report.relative_frobenius() - 2.0_f64.sqrt() / 6.0).abs() < 1e-14);
            let solution = factor.solve(&[6.0, 9.0][..]).unwrap();
            assert!((solution.as_slice()[0] - 1.0).abs() < 1e-14);
            assert!((solution.as_slice()[1] - 2.0).abs() < 1e-14);
            let saved = factor.l_matrix().clone();
            let permutation = factor.permutation().to_vec();
            let mut workspace = LinalgWorkspace::new();
            let interpretation = HermitianInput::Projected(HermitianProjectionPolicy::Within(Tolerance::new(0.0, 0.0).unwrap()));
            assert!($factor::factor_into(&input, interpretation, $($pivot,)? &options,
                &mut factor, &mut workspace, ExecutionContext::sequential()).is_err());
            assert_eq!(factor.l_matrix(), &saved);
            assert_eq!(factor.permutation(), &permutation);
            assert_eq!(factor.projection_report(), Some(&report));
            assert!($factor::factor_into(&input, HermitianInput::Exact, $($pivot,)? &options,
                &mut factor, &mut workspace, ExecutionContext::sequential()).is_err());
            assert_eq!(factor.projection_report(), Some(&report));
            $factor::factor_into(&input, HermitianInput::Triangle(Triangle::Lower),
                $($pivot,)? &options, &mut factor, &mut workspace, ExecutionContext::sequential()).unwrap();
            assert!(factor.projection_report().is_none());
            let exact = Matrix::from_vec(2, 2, vec![4.0, 1.0, 1.0, 4.0]).unwrap();
            $factor::factor_into(&exact, HermitianInput::Exact, $($pivot,)? &options,
                &mut factor, &mut workspace, ExecutionContext::sequential()).unwrap();
            assert!(factor.projection_report().is_none());

            for triangle in [Triangle::Lower, Triangle::Upper] {
                let (lower, upper) = if triangle == Triangle::Lower {
                    (Complex::new(1.0, -1.0), Complex::new(f64::NAN, f64::INFINITY))
                } else {
                    (Complex::new(f64::NAN, f64::INFINITY), Complex::new(1.0, 1.0))
                };
                let matrix = Matrix::from_vec(2, 2, vec![Complex::new(4.0, 0.0), upper, lower, Complex::new(4.0, 0.0)]).unwrap();
                let factor = $factor::factor_with_options(&matrix, HermitianInput::Triangle(triangle), $($pivot,)? &options).unwrap();
                assert!(factor.projection_report().is_none());
                assert!($factor::factor_with_options(&matrix, HermitianInput::Exact, $($pivot,)? &options).is_err());
                assert!($factor::factor_with_options(&matrix, HermitianInput::Projected(HermitianProjectionPolicy::Always), $($pivot,)? &options).is_err());
            }
            let imaginary_diagonal = Matrix::from_vec(1, 1, vec![Complex::new(4.0_f64, 2.0)]).unwrap();
            assert!($factor::factor_with_options(&imaginary_diagonal, HermitianInput::Triangle(Triangle::Lower), $($pivot,)? &options).is_err());
            let projected = $factor::factor_with_options(&imaginary_diagonal,
                HermitianInput::Projected(HermitianProjectionPolicy::Always), $($pivot,)? &options).unwrap();
            assert!((projected.projection_report().unwrap().absolute_frobenius() - 2.0).abs() < 1e-14);
            let empty = $factor::<f64>::factor_with_options(Matrix::<f64>::empty().view(), HermitianInput::Exact, $($pivot,)? &options).unwrap();
            assert!(empty.projection_report().is_none());
        }
    };
}
hermitian_contract!(
    cholesky_input_contract,
    CholeskyDecomposition,
    CholeskyPivoting::None
);
hermitian_contract!(ldl_input_contract, LdlDecomposition);

#[test]
fn svd_single_solve_handles_zero_and_positive_damping() {
    let options = LinalgExecutionOptions::new();
    let matrix = Matrix::from_vec(1, 2, vec![1.0_f64, 1.0]).unwrap();
    let full = SvdDecomposition::compute(&matrix).unwrap();
    let thin = ThinSvdDecomposition::compute(&matrix).unwrap();
    for damping in [0.0, 2.0] {
        let expected = 2.0 / (2.0 + damping);
        for x in [
            full.solve(&[2.0][..], RankPolicy::Exact, damping).unwrap(),
            thin.solve(&[2.0][..], RankPolicy::Exact, damping).unwrap(),
        ] {
            assert!(
                x.as_slice()
                    .iter()
                    .all(|&value| (value - expected).abs() < 1e-14)
            );
        }
    }
    let request = SvdWorkspaceRequest::solve(1, 2, &options);
    let mut workspace = SvdDecomposition::<f64>::workspace(&request).unwrap();
    for (damping, kind) in [
        (-1.0, scioxide_core::SciOxideErrorKind::InvalidInput),
        (f64::NAN, scioxide_core::SciOxideErrorKind::NonFinite),
        (f64::INFINITY, scioxide_core::SciOxideErrorKind::NonFinite),
    ] {
        let mut output = [81.0, 82.0, 83.0];
        assert_eq!(
            thin.solve_into(
                &[2.0][..],
                RankPolicy::Exact,
                damping,
                &scioxide_linalg::LinalgExecutionOptions::new(),
                VectorViewMut::from_slice_with_stride(&mut output, 2, 2).unwrap(),
                &mut workspace,
                ExecutionContext::sequential()
            )
            .unwrap_err()
            .kind(),
            kind
        );
        assert_eq!(output, [81.0, 82.0, 83.0]);
    }
}

#[test]
fn checked_cholesky_complex_permutation_and_numerical_failure() {
    let matrix = Matrix::from_vec(
        2,
        2,
        vec![
            Complex::new(2.0_f64, 0.0),
            Complex::new(1.0, 1.0),
            Complex::new(1.0, -1.0),
            Complex::new(5.0, 0.0),
        ],
    )
    .unwrap();
    let acceptance = DirectSolveOptions::new(RankPolicy::Exact)
        .with_residual(scioxide_linalg::Estimation::Estimate);
    let factor = CholeskyDecomposition::factor(
        &matrix,
        HermitianInput::Exact,
        CholeskyPivoting::Diagonal(scioxide_linalg::RankPolicy::DimensionScaledEpsilon),
    )
    .unwrap();
    assert_eq!(factor.permutation(), &[1, 0]);
    let result = factor
        .solve_with_options(
            &[Complex::new(5.0, 3.0), Complex::new(12.0, -5.0)][..],
            &acceptance,
        )
        .unwrap();
    let solution = result.solution().unwrap().as_slice();
    assert!((solution[0] - Complex::new(1.0, 1.0)).abs() < 1e-14);
    assert!((solution[1] - Complex::new(2.0, -1.0)).abs() < 1e-14);
    assert!(result.report().residual().is_some());

    let tiny = Matrix::from_vec(1, 1, vec![f64::MIN_POSITIVE]).unwrap();
    let factor =
        CholeskyDecomposition::factor(&tiny, HermitianInput::Exact, CholeskyPivoting::None)
            .unwrap();
    let mut output = [91.0];
    assert!(
        factor
            .solve_into(
                &[f64::MAX][..],
                &acceptance,
                VectorViewMut::from_slice(&mut output),
                &mut LinalgWorkspace::new(),
                ExecutionContext::sequential()
            )
            .is_err()
    );
    assert_eq!(output, [91.0]);
}

#[test]
fn cholesky_condition_preserves_extreme_nonzero_pivots() {
    let matrix = Matrix::from_vec(2, 2, vec![f64::MIN_POSITIVE, 0.0, 0.0, f64::MAX]).unwrap();
    let factor =
        CholeskyDecomposition::factor(&matrix, HermitianInput::Exact, CholeskyPivoting::None)
            .unwrap();
    assert!(factor.condition_estimate().unwrap().reciprocal().value() > 0.0);
    let matrix = Matrix::from_vec(2, 2, vec![f32::MIN_POSITIVE, 0.0, 0.0, f32::MAX]).unwrap();
    let factor =
        CholeskyDecomposition::factor(&matrix, HermitianInput::Exact, CholeskyPivoting::None)
            .unwrap();
    assert!(factor.condition_estimate().unwrap().reciprocal().value() > 0.0);
}

#[test]
fn square_factor_outputs_reject_wrong_shapes_without_mutation() {
    let matrix = Matrix::<f64>::identity(2).unwrap();
    let lu = scioxide_linalg::LuDecomposition::factor(&matrix).unwrap();
    let cholesky =
        CholeskyDecomposition::factor(&matrix, HermitianInput::Exact, CholeskyPivoting::None)
            .unwrap();
    for (rows, cols) in [(0, 0), (1, 2), (2, 1), (2, 3), (3, 2)] {
        let mut output = Matrix::from_vec(rows, cols, vec![71.0; rows * cols]).unwrap();
        for result in [
            lu.l_matrix_into(output.view_mut()),
            lu.u_matrix_into(output.view_mut()),
            scioxide_linalg::MatrixView::from(cholesky.l_matrix()).copy_into(
                scioxide_linalg::CopyPolicy::Preserve,
                &scioxide_linalg::LinalgExecutionOptions::new(),
                (output.view_mut()).reborrow(),
                scioxide_core::ExecutionContext::sequential(),
            ),
        ] {
            assert_eq!(
                result.unwrap_err().kind(),
                scioxide_core::SciOxideErrorKind::DimensionMismatch
            );
        }
        assert!(output.as_slice().iter().all(|&value| value == 71.0));
    }
    let mut output = [71.0; 6];
    lu.l_matrix_into(
        scioxide_linalg::MatrixViewMut::from_slice_with_strides(2, 2, &mut output, 3, 1).unwrap(),
    )
    .unwrap();
    assert_eq!(output, [1.0, 0.0, 71.0, 0.0, 1.0, 71.0]);
    scioxide_linalg::MatrixView::from(cholesky.l_matrix())
        .copy_into(
            scioxide_linalg::CopyPolicy::Preserve,
            &scioxide_linalg::LinalgExecutionOptions::new(),
            (scioxide_linalg::MatrixViewMut::from_slice_with_strides(2, 2, &mut output, 3, 1)
                .unwrap())
            .reborrow(),
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(output, [1.0, 0.0, 71.0, 0.0, 1.0, 71.0]);
}

#[test]
fn reported_qr_rank_is_the_truncation_rank() {
    use scioxide_core::RankThreshold;
    let options = LinalgExecutionOptions::new();
    let a = Matrix::from_vec(
        3,
        3,
        vec![1.0_f64, 0.0, 0.0, 0.0, 1e-3, 1e-3, 0.0, 0.0, 0.0],
    )
    .unwrap();
    let plan = HouseholderDecomposition::factor(
        &a,
        HouseholderOrientation::Qr,
        HouseholderPivoting::Column,
    )
    .unwrap()
    .into_minimum_norm(
        &(DirectSolveOptions::new(RankPolicy::Threshold(
            RankThreshold::new(1.2e-3, 0.0).unwrap(),
        )))
        .with_execution_options(options),
    )
    .unwrap();
    let result = plan.solve(&[1.0, 1.0, 0.0][..]).unwrap();
    assert_eq!(result.report().rank().rank(), 1);
    for (&actual, expected) in result
        .solution()
        .unwrap()
        .as_slice()
        .iter()
        .zip([1.0, 0.0, 0.0])
    {
        assert!((actual - expected).abs() < 1e-14);
    }
    assert!(result.report().residual().is_none());
}

#[test]
fn off_diagonal_growth_is_rejected_by_condition_gates() {
    let n = 60;
    let options = LinalgExecutionOptions::new();
    let acceptance = DirectSolveOptions::new(RankPolicy::Exact)
        .with_minimum_reciprocal_condition(ReciprocalConditionThreshold::new(1e-8).unwrap());
    let a = Matrix::from_vec(
        n,
        n,
        (0..n)
            .flat_map(|i| {
                (0..n).map(move |j| {
                    if i == j {
                        1.0_f64
                    } else if i < j {
                        -1.0
                    } else {
                        0.0
                    }
                })
            })
            .collect(),
    )
    .unwrap();
    let rhs = vec![1.0; n];
    let lu = scioxide_linalg::LuDecomposition::factor(&a).unwrap();
    let estimate = lu.condition_estimate().unwrap().reciprocal().value();
    let exact = 1.0 / (n as f64 * 2.0_f64.powi(n as i32 - 1));
    assert!((estimate / exact - 1.0).abs() < 1e-12);
    assert_eq!(
        lu.solve_with_options(&rhs[..], &acceptance)
            .unwrap()
            .report()
            .termination(),
        DirectSolveTermination::IllConditioned
    );
    let qr =
        HouseholderDecomposition::factor(&a, HouseholderOrientation::Qr, HouseholderPivoting::None)
            .unwrap();
    assert_eq!(
        qr.solve_with_options(&rhs[..], &(acceptance).with_execution_options(options))
            .unwrap()
            .report()
            .termination(),
        DirectSolveTermination::IllConditioned
    );
    // Cholesky/LDL require Hermitian inputs. Tᴴ T has exact integer entries;
    // its unit-diagonal Cholesky factor is representable despite extreme conditioning.
    let gram = Matrix::from_vec(
        n,
        n,
        (0..n)
            .flat_map(|i| {
                (0..n).map(move |j| {
                    if i == j {
                        (i + 1) as f64
                    } else {
                        i.min(j) as f64 - 1.0
                    }
                })
            })
            .collect(),
    )
    .unwrap();
    let cholesky =
        CholeskyDecomposition::factor(&gram, HermitianInput::Exact, CholeskyPivoting::None)
            .unwrap();
    assert_eq!(
        cholesky
            .solve_with_options(&rhs[..], &acceptance)
            .unwrap()
            .report()
            .termination(),
        DirectSolveTermination::IllConditioned
    );
    let ldl = LdlDecomposition::factor(&gram, HermitianInput::Exact).unwrap();
    assert_eq!(
        ldl.solve_with_options(&rhs[..], &acceptance)
            .unwrap()
            .report()
            .termination(),
        DirectSolveTermination::IllConditioned
    );
}

fn numerical_gram_rank<S: NumericScalar>() {
    let mut state = 1188_u64;
    let mut sample = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        S::Real::try_from_f64_lossy(((state >> 32) % 1000) as f64 / 499.0 - 1.0).unwrap()
    };
    for (n, r) in (0..50)
        .map(|_| (8, 3))
        .chain([(8, 0), (8, 1), (8, 7), (131, 129)])
    {
        let x: Vec<S> = (0..n * r)
            .map(|k| {
                let diagonal = if k / r == k % r {
                    S::Real::TWO
                } else {
                    S::Real::ZERO
                };
                let re = sample() / S::Real::try_from_usize_exact(16).unwrap() + diagonal;
                let im = if S::IS_REAL {
                    S::Real::ZERO
                } else {
                    sample() / S::Real::try_from_usize_exact(16).unwrap()
                };
                S::from_components(re, im)
            })
            .collect();
        let mut a = Matrix::zeros(n, n).unwrap();
        for i in 0..n {
            for j in 0..=i {
                let value = (0..r).fold(S::ZERO, |sum, k| sum + x[i * r + k] * x[j * r + k].conj());
                a.as_mut_slice()[i * n + j] = value;
                a.as_mut_slice()[j * n + i] = value.conj();
            }
        }
        let factor = CholeskyDecomposition::factor(
            &a,
            HermitianInput::Exact,
            CholeskyPivoting::Diagonal(RankPolicy::DimensionScaledEpsilon),
        )
        .unwrap();
        let report = factor.definiteness_report(RankPolicy::Exact).unwrap();
        assert_eq!(report.factor_rank().rank(), r, "{n}x{n}");
        assert_eq!(report.effective_rank().rank(), r);
        let norm = a
            .as_slice()
            .iter()
            .fold(0.0_f64, |m, &x| m.max(x.abs().into_f64()));
        let bound = 64.0 * n as f64 * S::Real::EPSILON.into_f64() * norm.max(1.0);
        for i in 0..n {
            for j in 0..n {
                let value = (0..r).fold(S::ZERO, |sum, k| {
                    sum + factor.l_matrix().as_slice()[i * n + k]
                        * factor.l_matrix().as_slice()[j * n + k].conj()
                });
                let wanted = a.as_slice()[factor.permutation()[i] * n + factor.permutation()[j]];
                assert!(
                    (value - wanted).abs().into_f64() <= bound,
                    "rank {r}, entry ({i},{j})"
                );
            }
        }
    }
}
#[test]
fn gram_rank_f32() {
    numerical_gram_rank::<f32>();
}
#[test]
fn gram_rank_f64() {
    numerical_gram_rank::<f64>();
}
#[test]
fn gram_rank_complex32() {
    numerical_gram_rank::<Complex<f32>>();
}
#[test]
fn gram_rank_complex64() {
    numerical_gram_rank::<Complex<f64>>();
}

#[test]
fn pivoted_cholesky_stops_at_roundoff_but_rejects_indefinite_blocks() {
    let pivoting = CholeskyPivoting::Diagonal(RankPolicy::DimensionScaledEpsilon);
    let a = Matrix::from_vec(2, 2, vec![1.0_f64, 1.0, 1.0, 1.0 - 1e-15]).unwrap();
    let factor = CholeskyDecomposition::factor(&a, HermitianInput::Exact, pivoting).unwrap();
    assert_eq!(
        factor
            .definiteness_report(RankPolicy::Exact)
            .unwrap()
            .factor_rank()
            .rank(),
        1
    );
    for values in [
        [1.0, 2.0, 2.0, 1.0],
        [0.0, 1.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, -1.0],
    ] {
        let a = Matrix::from_vec(2, 2, values.to_vec()).unwrap();
        assert_eq!(
            CholeskyDecomposition::factor(&a, HermitianInput::Exact, pivoting)
                .unwrap_err()
                .kind(),
            scioxide_core::SciOxideErrorKind::DomainError
        );
    }
    let a = Matrix::from_vec(2, 2, vec![1.0_f64, 0.0, 0.0, 1e-8]).unwrap();
    let policy = RankPolicy::Threshold(scioxide_core::RankThreshold::new(1e-7, 0.0).unwrap());
    let factor = CholeskyDecomposition::factor(
        &a,
        HermitianInput::Exact,
        CholeskyPivoting::Diagonal(policy),
    )
    .unwrap();
    let report = factor
        .definiteness_report(RankPolicy::Exact)
        .unwrap()
        .factor_rank();
    assert_eq!(report.rank(), 1);
    assert!((report.threshold() - 1e-7).abs() < 1e-20);
}

#[test]
fn coarse_cholesky_rank_policy_does_not_hide_indefiniteness() {
    let a = Matrix::from_vec(
        3,
        3,
        vec![1.0_f64, -0.6, -0.6, -0.6, 1.0, -0.6, -0.6, -0.6, 1.0],
    )
    .unwrap();
    let policy = RankPolicy::Threshold(scioxide_core::RankThreshold::new(2.0, 0.0).unwrap());
    assert_eq!(
        CholeskyDecomposition::factor(
            &a,
            HermitianInput::Exact,
            CholeskyPivoting::Diagonal(policy)
        )
        .unwrap_err()
        .kind(),
        scioxide_core::SciOxideErrorKind::DomainError
    );
}

#[test]
fn structured_residuals_share_the_dense_backward_error() {
    use scioxide_core::Accuracy;
    use scioxide_linalg::{BandedLuDecomposition, BandedMatrix, Estimation, StructuredOperation};
    let nan = f64::NAN;
    // Column-major band storage; NaN padding lies outside the band.
    let band = BandedMatrix::new(
        4,
        4,
        1,
        2,
        4,
        vec![
            nan, nan, 4.0, 2.0, nan, 1.0, 5.0, -1.0, 0.5, 1.0, 6.0, 3.0, 0.25, 2.0, 7.0, nan,
        ],
    )
    .unwrap();
    let dense = [
        [4.0, 1.0, 0.5, 0.0],
        [2.0, 5.0, 1.0, 0.25],
        [0.0, -1.0, 6.0, 2.0],
        [0.0, 0.0, 3.0, 7.0],
    ];
    let rhs = Matrix::from_vec(4, 1, vec![0.1, -0.7, 0.3, 0.9]).unwrap();
    let maximum = |values: &[f64]| values.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    for accuracy in [Accuracy::Standard, Accuracy::Exact] {
        let options = LinalgExecutionOptions::new().with_accuracy(accuracy);
        let factor = BandedLuDecomposition::factor_with_options(&band, &options).unwrap();
        let acceptance = DirectSolveOptions::new(RankPolicy::Exact)
            .with_residual(Estimation::Estimate)
            .with_execution_options(options);
        for operation in [
            StructuredOperation::Normal,
            StructuredOperation::Transpose,
            StructuredOperation::Adjoint,
        ] {
            let result = factor
                .solve_matrix_with_options(operation, &rhs, &acceptance)
                .unwrap();
            let x = result.solution().unwrap().as_slice();
            let op = |i: usize, j: usize| {
                if operation == StructuredOperation::Normal {
                    dense[i][j]
                } else {
                    dense[j][i]
                }
            };
            let norm = (0..4)
                .map(|i| (0..4).map(|j| op(i, j).abs()).sum::<f64>())
                .fold(0.0_f64, f64::max);
            let denominator = norm * maximum(x) + maximum(rhs.as_slice());
            let expected = (0..4)
                .map(|i| rhs.as_slice()[i] - (0..4).map(|j| op(i, j) * x[j]).sum::<f64>())
                .fold(0.0_f64, |m, r| m.max(r.abs()));
            let residual = result.report().residual().unwrap();
            assert_eq!(
                residual.scaling().rule(),
                scioxide_core::ResidualScalingRule::BackwardError
            );
            assert_eq!(residual.norm(), scioxide_core::NormSelection::Infinity);
            assert!(
                (residual.scaling().denominator() - denominator).abs()
                    <= 8.0 * f64::EPSILON * denominator
            );
            assert!((residual.raw_norm() - expected).abs() <= 8.0 * f64::EPSILON * denominator);
        }
    }
}

#[test]
fn triangular_and_banded_gates_detect_off_diagonal_growth() {
    use scioxide_core::{DiagonalKind, Triangle};
    use scioxide_linalg::{
        BandedLuDecomposition, BandedMatrix, StructuredOperation, TriangularPlan,
    };
    let n = 60;
    let entries: Vec<f64> = (0..n)
        .flat_map(|i| {
            (0..n).map(move |j| {
                if i == j {
                    1.0
                } else if i < j {
                    -1.0
                } else {
                    0.0
                }
            })
        })
        .collect();
    let a = Matrix::from_vec(n, n, entries).unwrap();
    let triangular =
        TriangularPlan::from_dense(a.view(), Triangle::Upper, DiagonalKind::Unit).unwrap();
    let mut band = vec![0.0; n * n];
    for col in 0..n {
        for row in 0..=col {
            band[col * n + n - 1 + row - col] = a.as_slice()[row * n + col];
        }
    }
    let band = BandedMatrix::new(n, n, 0, n - 1, n, band).unwrap();
    let factor = BandedLuDecomposition::factor(&band).unwrap();
    let exact = 1.0 / (n as f64 * 2.0_f64.powi(n as i32 - 1));
    let acceptance = DirectSolveOptions::new(RankPolicy::Exact)
        .with_minimum_reciprocal_condition(ReciprocalConditionThreshold::new(1e-8).unwrap());
    let rhs = vec![1.0; n];
    for (estimate, report) in [
        (
            triangular.condition_estimate().unwrap(),
            *triangular
                .solve_with_options(StructuredOperation::Normal, &rhs[..], &acceptance)
                .unwrap()
                .report(),
        ),
        (
            factor.condition_estimate().unwrap(),
            *factor
                .solve_with_options(StructuredOperation::Normal, &rhs[..], &acceptance)
                .unwrap()
                .report(),
        ),
    ] {
        assert!((estimate.reciprocal().value() / exact - 1.0).abs() < 1e-12);
        assert_eq!(report.termination(), DirectSolveTermination::IllConditioned);
    }
}

fn structured_complex_pivot_beyond_real_magnitude<S: NumericScalar>() {
    use scioxide_core::{DiagonalKind, Triangle};
    use scioxide_linalg::{
        BandedLuDecomposition, BandedMatrix, StructuredOperation, TriangularPlan,
        TridiagonalLuDecomposition, TridiagonalMatrixView,
    };
    let part = S::Real::MAX * S::Real::try_from_f64_lossy(0.75).unwrap();
    let pivot = S::from_components(part, part);
    let matrix = Matrix::from_vec(1, 1, vec![pivot]).unwrap();
    let band = BandedMatrix::new(1, 1, 0, 0, 1, vec![pivot]).unwrap();
    let diagonal = [pivot];
    let tri = TridiagonalMatrixView::new(&[], &diagonal, &[]).unwrap();
    let acceptance = DirectSolveOptions::new(RankPolicy::Exact).with_minimum_reciprocal_condition(
        ReciprocalConditionThreshold::new(S::Real::try_from_f64_lossy(0.5).unwrap()).unwrap(),
    );
    let results = [
        TriangularPlan::from_dense(matrix.view(), Triangle::Upper, DiagonalKind::Explicit)
            .unwrap()
            .solve_with_options(StructuredOperation::Normal, &diagonal[..], &acceptance)
            .unwrap(),
        BandedLuDecomposition::factor(&band)
            .unwrap()
            .solve_with_options(StructuredOperation::Normal, &diagonal[..], &acceptance)
            .unwrap(),
        TridiagonalLuDecomposition::factor(tri)
            .unwrap()
            .solve_with_options(StructuredOperation::Normal, &diagonal[..], &acceptance)
            .unwrap(),
    ];
    for result in results {
        assert_eq!(
            result.report().termination(),
            DirectSolveTermination::Accepted
        );
        assert_eq!(result.report().rank().rank(), 1);
        assert_eq!(result.report().rank().scale(), S::Real::MAX);
        let error = (result.solution().unwrap().as_slice()[0] - S::ONE).abs();
        assert!(error <= S::Real::EPSILON * S::Real::try_from_f64_lossy(8.0).unwrap());
    }
}

#[test]
fn structured_exact_rank_preserves_extreme_complex_solves() {
    structured_complex_pivot_beyond_real_magnitude::<Complex<f32>>();
    structured_complex_pivot_beyond_real_magnitude::<Complex<f64>>();
}

#[test]
fn hermitian_projection_measures_norms_beyond_the_scalar_range() {
    use scioxide_linalg::{
        CholeskyDecomposition, CholeskyPivoting, HermitianInput, HermitianProjectionPolicy,
        LdlDecomposition,
    };
    // ||A||_F exceeds MAX although every entry and the factor are representable.
    let big = 0.9 * f64::MAX;
    let a = Matrix::from_vec(2, 2, vec![big, 0.0, 0.0, big]).unwrap();
    let projected = HermitianInput::Projected(HermitianProjectionPolicy::Always);
    let factor = CholeskyDecomposition::factor(&a, projected, CholeskyPivoting::None).unwrap();
    let report = factor.projection_report().unwrap();
    assert_eq!(report.absolute_frobenius(), 0.0);
    assert_eq!(report.relative_frobenius(), 0.0);
    // A huge complex entry has a finite projection change measured componentwise.
    let z = scioxide_core::Complex64::new(f64::MAX / 1.2, f64::MAX / 1.2);
    let one = scioxide_core::Complex64::new(1.0, 0.0);
    let zero = scioxide_core::Complex64::new(0.0, 0.0);
    let c = Matrix::from_vec(2, 2, vec![one, z, zero, one]).unwrap();
    let ldl = LdlDecomposition::factor(&c, projected);
    assert!(
        ldl.as_ref()
            .map(|f| f.projection_report().is_some())
            .unwrap_or_else(|error| error.kind() != scioxide_core::SciOxideErrorKind::NonFinite),
        "{ldl:?}"
    );
}

#[test]
fn exact_lu_vector_and_matrix_solves_share_arithmetic() {
    use scioxide_core::Accuracy;
    use scioxide_linalg::{
        DirectSolveOptions, LinalgExecutionOptions, LinalgWorkspace, LuDecomposition, MatrixView,
        RankPolicy, VectorView, VectorViewMut,
    };
    let n = 6;
    let a = Matrix::from_vec(
        n,
        n,
        (0..n * n)
            .map(|i| ((i * 37 % 17) as f64 - 8.0) / 3.0 + if i % (n + 1) == 0 { 9.0 } else { 0.0 })
            .collect(),
    )
    .unwrap();
    let b: Vec<f64> = (0..n)
        .map(|i| 1e8 * (i as f64 - 2.5) + 1.0 / (i + 1) as f64)
        .collect();
    let execution = LinalgExecutionOptions::new().with_accuracy(Accuracy::Exact);
    let factor = LuDecomposition::factor_with_options(&a, &execution).unwrap();
    let options = DirectSolveOptions::new(RankPolicy::Exact).with_execution_options(execution);
    let mut vector = vec![0.0; n];
    let _ = factor
        .solve_into(
            VectorView::from_slice(&b),
            &options,
            VectorViewMut::from_slice(&mut vector),
            &mut LinalgWorkspace::new(),
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap();
    let mut matrix = Matrix::zeros(n, 1).unwrap();
    let _ = factor
        .solve_matrix_into(
            MatrixView::from_slice(n, 1, &b).unwrap(),
            &options,
            matrix.view_mut(),
            &mut LinalgWorkspace::new(),
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(vector.as_slice(), matrix.as_slice());
}

#[test]
fn pivoted_cholesky_stops_at_rounding_floor_under_exact_policy() {
    // G = B Bᵀ is exactly positive semidefinite with rank 2. Its Schur
    // complement after two steps is rounding noise, which every policy discards.
    let b = [
        [-2.0, -7.0],
        [-5.0, 7.0],
        [5.0, 7.0],
        [-1.0, 8.0],
        [-1.0, -1.0],
        [1.0, 5.0],
        [3.0, 8.0],
        [-5.0, 7.0],
        [4.0, 7.0],
    ];
    let n = b.len();
    let g: Vec<f64> = (0..n * n)
        .map(|k| b[k / n][0] * b[k % n][0] + b[k / n][1] * b[k % n][1])
        .collect();
    let g = Matrix::from_vec(n, n, g).unwrap();
    for policy in [RankPolicy::Exact, RankPolicy::DimensionScaledEpsilon] {
        let factor = CholeskyDecomposition::factor(
            &g,
            HermitianInput::Exact,
            CholeskyPivoting::Diagonal(policy),
        )
        .unwrap();
        let report = factor
            .definiteness_report(RankPolicy::Exact)
            .unwrap()
            .factor_rank();
        assert_eq!(report.rank(), 2);
        // τ₀ = n ε max_i g_ii with max_i g_ii = 74.
        assert!((report.threshold() / (9.0 * f64::EPSILON * 74.0) - 1.0).abs() < 1e-12);
        let l = factor.l_matrix().view();
        let p = factor.permutation();
        for i in 0..n {
            for j in 0..n {
                let product: f64 = (0..n)
                    .map(|k| l.get(i, k).unwrap() * l.get(j, k).unwrap())
                    .sum();
                assert!((product - g.view().get(p[i], p[j]).unwrap()).abs() < 1e-12 * 74.0);
            }
        }
    }
}

#[test]
fn cholesky_overflowing_scaled_entry_is_indefinite() {
    // |a_01| > √(a_00 a_11), and the equilibrated a_01 exceeds the range.
    let a = Matrix::from_vec(2, 2, vec![1e-300_f64, 1e300, 1e300, 1.0]).unwrap();
    for pivoting in [
        CholeskyPivoting::None,
        CholeskyPivoting::Diagonal(RankPolicy::DimensionScaledEpsilon),
    ] {
        assert_eq!(
            CholeskyDecomposition::factor(&a, HermitianInput::Exact, pivoting)
                .unwrap_err()
                .kind(),
            scioxide_core::SciOxideErrorKind::DomainError
        );
    }
}

#[test]
fn ldl_keeps_subnormal_matrices_and_multipliers_accurate() {
    // A = 2^-1070 M with M x = e₁ for x = (0.8125, 0.5, 1.375).
    let scale = 2.0_f64.powi(-535) * 2.0_f64.powi(-535);
    let base = [4.0, 1.0, -2.0, 1.0, -3.0, 0.5, -2.0, 0.5, 1.0];
    let a = Matrix::from_vec(3, 3, base.iter().map(|v| v * scale).collect()).unwrap();
    let factor = LdlDecomposition::factor(&a, HermitianInput::Exact).unwrap();
    let x = factor.solve(&[scale, 0.0, 0.0][..]).unwrap();
    for (computed, exact) in x.as_slice().iter().zip([0.8125, 0.5, 1.375]) {
        assert!((computed - exact).abs() < 1e-14);
    }
    let inertia = factor.inertia(RankPolicy::Exact).unwrap();
    assert_eq!((inertia.positive(), inertia.negative()), (2, 1));
    // det M = −4 by cofactor expansion, and det A = 2^-3210 det M.
    let det_m = 4.0 * (-3.0 * 1.0 - 0.5 * 0.5) - 1.0 * (1.0 * 1.0 - 0.5 * -2.0)
        + -2.0 * (1.0 * 0.5 - -3.0 * -2.0);
    let slogdet = factor.slogdet().unwrap();
    assert_eq!(slogdet.phase(), det_m.signum());
    let expected = det_m.abs().ln() - 3210.0 * 2.0_f64.ln();
    assert!((slogdet.log_abs() - expected).abs() < 1e-12 * expected.abs());
    // Gradual underflow of a multiplier or a solution component is not an error.
    let big = 2.0_f64.powi(1000);
    let spd = Matrix::from_vec(2, 2, vec![big, 1.0 / big, 1.0 / big, 1.0]).unwrap();
    let factor = LdlDecomposition::factor(&spd, HermitianInput::Exact).unwrap();
    assert_eq!(factor.diagonal()[1], 1.0);
    let swap = Matrix::from_vec(2, 2, vec![0.0, big, big, 0.0]).unwrap();
    let x = LdlDecomposition::factor(&swap, HermitianInput::Exact)
        .unwrap()
        .solve(&[2.0_f64.powi(-100), 1.0][..])
        .unwrap();
    assert_eq!(x.as_slice(), &[1.0 / big, 0.0]);
}

#[test]
fn ldl_exact_solves_do_not_depend_on_the_kernel() {
    let n = 48;
    let entry = |i: usize, j: usize| {
        let (i, j) = (i.max(j) as f64, i.min(j) as f64);
        ((i * 7.0 + j * 3.0) % 11.0 - 5.0) / (1.0 + i + j)
    };
    let a = Matrix::from_vec(n, n, (0..n * n).map(|k| entry(k / n, k % n)).collect()).unwrap();
    let b: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37).sin()).collect();
    let exact = LinalgExecutionOptions::new().with_accuracy(scioxide_core::Accuracy::Exact);
    let options = DirectSolveOptions::new(RankPolicy::Exact).with_execution_options(exact);
    let factor = LdlDecomposition::factor_with_options(&a, HermitianInput::Exact, &exact).unwrap();
    let solve = |context: ExecutionContext<'static>| {
        let mut x = vec![0.0; n];
        let report = factor
            .solve_into(
                &b[..],
                &options,
                VectorViewMut::from_slice(&mut x),
                &mut LinalgWorkspace::new(),
                context,
            )
            .unwrap();
        assert_eq!(report.termination(), DirectSolveTermination::Accepted);
        x
    };
    let portable = solve(ExecutionContext::sequential());
    let native = solve(ExecutionContext::sequential_native());
    assert!(
        portable
            .iter()
            .zip(&native)
            .all(|(p, q)| p.to_bits() == q.to_bits())
    );
    let standard = factor.solve(&b[..]).unwrap();
    for (x, y) in portable.iter().zip(standard.as_slice()) {
        assert!((x - y).abs() < 1e-10 * (1.0 + y.abs()));
    }
}

#[test]
fn lu_unscaling_overflow_is_numerical_instability() {
    // The tiny second column has column scale 2^1000, so x₂ = 2^1000 · 1e300.
    let tiny = 2.0_f64.powi(-1000);
    let a = Matrix::from_vec(2, 2, vec![1.0, tiny, 1.0, 0.0]).unwrap();
    let lu = scioxide_linalg::LuDecomposition::factor(&a).unwrap();
    let is_overflow = |error: scioxide_core::SciOxideError| {
        matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::NumericalInstability {
                issue: scioxide_core::NumericalIssue::Overflow,
                ..
            }
        )
    };
    assert!(is_overflow(lu.solve(&[1e300, 0.0][..]).unwrap_err()));
    let rhs = Matrix::from_vec(2, 1, vec![1e300, 0.0]).unwrap();
    assert!(is_overflow(lu.solve_matrix(&rhs).unwrap_err()));
    let x = lu.solve(&[1.0, 0.5][..]).unwrap();
    assert!((x.as_slice()[0] - 0.5).abs() < 1e-15);
    assert!((x.as_slice()[1] * tiny - 0.5).abs() < 1e-15);
}

#[test]
fn householder_minimum_norm_with_zero_diagonals() {
    fn check<S: NumericScalar>() {
        svd_cases::zero_diagonals::<S>(|name, a| {
            let options = LinalgExecutionOptions::new();
            let acceptance = DirectSolveOptions::new(RankPolicy::DimensionScaledEpsilon);
            let bound = 512.0 * a.rows().max(a.cols()) as f64 * S::Real::EPSILON.into_f64();
            // x = A^H y belongs to the row space, so it is the unique
            // minimum-norm solution of A x = b, independently of any SVD.
            let x: Vec<S> = (0..a.cols())
                .map(|j| {
                    (0..a.rows()).fold(S::ZERO, |sum, i| {
                        sum + a.as_slice()[i * a.cols() + j].conj()
                    })
                })
                .collect();
            let b: Vec<S> = (0..a.rows())
                .map(|i| {
                    a.as_slice()[i * a.cols()..(i + 1) * a.cols()]
                        .iter()
                        .zip(&x)
                        .fold(S::ZERO, |sum, (&a, &x)| sum + a * x)
                })
                .collect();
            for orientation in [HouseholderOrientation::Qr, HouseholderOrientation::Lq] {
                for pivoting in [HouseholderPivoting::None, HouseholderPivoting::Column] {
                    let factor =
                        HouseholderDecomposition::factor(&a, orientation, pivoting).unwrap();
                    if pivoting == HouseholderPivoting::None {
                        assert_eq!(
                            factor
                                .into_minimum_norm(&(acceptance).with_execution_options(options))
                                .unwrap_err()
                                .kind(),
                            scioxide_core::SciOxideErrorKind::InvalidInput
                        );
                        continue;
                    }
                    let plan = factor
                        .into_minimum_norm(&(acceptance).with_execution_options(options))
                        .unwrap();
                    let outcome = plan.solve(&b[..]).unwrap_or_else(|error| {
                        panic!(
                            "{name}, {}x{}, {}, {orientation:?}, {pivoting:?}: {error}",
                            a.rows(),
                            a.cols(),
                            core::any::type_name::<S>()
                        )
                    });
                    assert_eq!(
                        outcome.report().termination(),
                        DirectSolveTermination::Accepted,
                        "{name}"
                    );
                    let actual = outcome.solution().unwrap().as_slice();
                    for (&actual, &expected) in actual.iter().zip(&x) {
                        assert!(
                            (actual - expected).abs().into_f64() <= bound,
                            "{name}: {actual:?} != {expected:?}"
                        );
                    }
                    for (row, &expected) in a.as_slice().chunks_exact(a.cols()).zip(&b) {
                        let value = row
                            .iter()
                            .zip(actual)
                            .fold(S::ZERO, |sum, (&a, &x)| sum + a * x);
                        assert!((value - expected).abs().into_f64() <= bound, "{name}");
                    }
                }
            }
        });
    }
    check::<f32>();
    check::<f64>();
    check::<Complex<f32>>();
    check::<Complex<f64>>();
}

#[test]
fn empty_direct_factors_share_identity_and_undefined_condition() {
    use scioxide_linalg::{
        BandedCholeskyDecomposition, BandedLuDecomposition, LuDecomposition, MatrixView,
        StructuredOperation, TriangularPlan, TridiagonalLuDecomposition,
    };
    let policy = DirectSolveOptions::new(RankPolicy::Exact);
    macro_rules! check {
        ($policy:expr, $gated:expr, $factor:expr $(, $op:expr)?) => {{
            let (policy, gated) = ($policy, $gated);
            let factor = $factor;
            let result = factor.solve_with_options($($op,)? &[][..], &policy).unwrap();
            assert_eq!(result.report().termination(), DirectSolveTermination::Accepted);
            assert_eq!(result.report().rank().rank(), 0);
            assert!(result.report().condition().is_none());
            assert!(result.solution().unwrap().is_empty());
            assert_eq!(result.report().residual().map(|r| r.scaled_norm()), (policy.residual() == scioxide_linalg::Estimation::Estimate).then_some(0.0));
            let columns = factor.solve_matrix_with_options($($op,)? MatrixView::from_slice(0, usize::MAX, &[]).unwrap(), &policy).unwrap();
            assert_eq!(columns.solution().unwrap().cols(), usize::MAX);
            assert_eq!(factor.condition_estimate().unwrap_err().kind(), scioxide_core::SciOxideErrorKind::DomainError);
            assert_eq!(factor.solve_with_options($($op,)? &[][..], &gated).unwrap_err().kind(), scioxide_core::SciOxideErrorKind::DomainError);
        }};
    }
    for policy in [
        policy,
        policy.with_residual(scioxide_linalg::Estimation::Estimate),
        DirectSolveOptions::new(RankPolicy::Threshold(
            scioxide_core::RankThreshold::new(1.0, 0.0).unwrap(),
        ))
        .with_residual(scioxide_linalg::Estimation::Estimate),
    ] {
        let gated = policy
            .with_minimum_reciprocal_condition(ReciprocalConditionThreshold::new(0.0).unwrap());
        check!(policy, gated, LuDecomposition::<f64>::empty());
        check!(policy, gated, CholeskyDecomposition::<f64>::empty());
        check!(policy, gated, LdlDecomposition::<f64>::empty());
        check!(policy, gated, HouseholderDecomposition::<f64>::empty());
        check!(
            policy,
            gated,
            BandedLuDecomposition::<f64>::empty(),
            StructuredOperation::Normal
        );
        check!(
            policy,
            gated,
            BandedCholeskyDecomposition::<f64>::empty(),
            StructuredOperation::Normal
        );
        check!(
            policy,
            gated,
            TridiagonalLuDecomposition::<f64>::empty(),
            StructuredOperation::Normal
        );
        check!(
            policy,
            gated,
            TriangularPlan::<f64>::from_dense(
                MatrixView::from_slice(0, 0, &[]).unwrap(),
                scioxide_core::Triangle::Lower,
                scioxide_core::DiagonalKind::Unit
            )
            .unwrap(),
            StructuredOperation::Normal
        );
    }
    let plan = HouseholderDecomposition::<f64>::empty()
        .into_minimum_norm(&policy)
        .unwrap();
    assert!(plan.solve(&[][..]).unwrap().solution().unwrap().is_empty());
    assert!(plan.condition_estimate().is_err());
    for value in [
        LuDecomposition::<f64>::empty().slogdet().unwrap(),
        CholeskyDecomposition::<f64>::empty().slogdet().unwrap(),
        LdlDecomposition::<f64>::empty().slogdet().unwrap(),
    ] {
        assert_eq!(value.phase(), 1.0);
        assert_eq!(value.log_abs(), 0.0);
    }
}
