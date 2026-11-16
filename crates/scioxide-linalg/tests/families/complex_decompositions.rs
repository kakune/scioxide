//! Complex direct-factorization reconstruction and solve coverage.

#![allow(clippy::panic)]

use scioxide_core::Complex;
use scioxide_linalg::{
    CholeskyDecomposition, DirectSolveTermination, HouseholderDecomposition,
    HouseholderOrientation, LuDecomposition, Matrix, RankPolicy, Vector,
};

macro_rules! complex_factorization_suite {
    ($name:ident, $real:ty, $tolerance:expr) => {
        #[test]
        fn $name() {
            type Scalar = Complex<$real>;
            let c = |real, imaginary| Scalar::new(real, imaginary);
            let tolerance: $real = $tolerance;

            let lu_matrix = Matrix::from_vec(
                3,
                3,
                vec![
                    c(0.0, 0.0),
                    c(2.0, 1.0),
                    c(1.0, -1.0),
                    c(4.0, -1.0),
                    c(1.0, 0.5),
                    c(0.5, 0.25),
                    c(1.0, 0.5),
                    c(-0.5, 1.0),
                    c(5.0, 0.0),
                ],
            )
            .unwrap_or_else(|error| panic!("{error}"));
            let expected = Vector::from_vec(vec![c(1.0, -0.5), c(-0.25, 1.0), c(0.75, 0.5)]);
            let rhs = lu_matrix
                .view()
                .mul_vec(&expected)
                .unwrap_or_else(|error| panic!("{error}"));
            let lu = LuDecomposition::factor(&lu_matrix).unwrap_or_else(|error| panic!("{error}"));
            assert_ne!(lu.permutation(), &[0, 1, 2]);
            let solution = lu.solve(&rhs).unwrap_or_else(|error| panic!("{error}"));
            for (&actual, &wanted) in solution.as_slice().iter().zip(expected.as_slice()) {
                assert!((actual - wanted).norm() < tolerance);
            }
            let lu_product = lu
                .to_l_matrix()
                .and_then(|lower| lower.view().mul_mat(&lu.to_u_matrix()?))
                .unwrap_or_else(|error| panic!("{error}"));
            for row in 0..3 {
                for col in 0..3 {
                    let actual = lu_product
                        .view()
                        .get(row, col)
                        .unwrap_or_else(|error| panic!("{error}"));
                    let wanted = lu_matrix
                        .view()
                        .get(lu.permutation()[row], col)
                        .unwrap_or_else(|error| panic!("{error}"));
                    assert!((actual - wanted).norm() < tolerance);
                }
            }

            let qr_matrix = Matrix::from_vec(
                4,
                3,
                vec![
                    c(2.0, 1.0),
                    c(1.0, -1.0),
                    c(0.5, 0.0),
                    c(-1.0, 0.5),
                    c(3.0, 0.0),
                    c(1.0, 1.0),
                    c(0.5, -1.0),
                    c(1.0, 0.25),
                    c(4.0, -0.5),
                    c(1.0, 0.0),
                    c(-0.5, 0.75),
                    c(2.0, 0.5),
                ],
            )
            .unwrap_or_else(|error| panic!("{error}"));
            let qr = HouseholderDecomposition::factor(
                &qr_matrix,
                HouseholderOrientation::Qr,
                scioxide_linalg::HouseholderPivoting::Column,
            )
            .unwrap_or_else(|error| panic!("{error}"));
            let q = qr.to_q_matrix().unwrap_or_else(|error| panic!("{error}"));
            let reconstructed = qr.reconstruct().unwrap_or_else(|error| panic!("{error}"));
            for (&actual, &wanted) in reconstructed.as_slice().iter().zip(qr_matrix.as_slice()) {
                assert!((actual - wanted).norm() < tolerance * 4.0);
            }
            let unitary = q
                .view()
                .adjoint()
                .try_to_matrix()
                .and_then(|adjoint| adjoint.view().mul_mat(&q))
                .unwrap_or_else(|error| panic!("{error}"));
            for row in 0..4 {
                for col in 0..4 {
                    let wanted = if row == col { c(1.0, 0.0) } else { c(0.0, 0.0) };
                    let actual = unitary
                        .view()
                        .get(row, col)
                        .unwrap_or_else(|error| panic!("{error}"));
                    assert!((actual - wanted).norm() < tolerance * 4.0);
                }
            }
            let qr_expected = Vector::from_vec(vec![c(1.0, 0.5), c(-0.5, 1.0), c(0.25, -0.5)]);
            let qr_rhs = qr_matrix
                .view()
                .mul_vec(&qr_expected)
                .unwrap_or_else(|error| panic!("{error}"));
            let qr_outcome = qr
                .into_minimum_norm(
                    &((scioxide_linalg::DirectSolveOptions::new(
                        RankPolicy::DimensionScaledEpsilon,
                    ))
                    .with_execution_options(scioxide_linalg::LinalgExecutionOptions::new()))
                    .with_residual(scioxide_linalg::Estimation::Estimate),
                )
                .unwrap()
                .solve(&qr_rhs)
                .unwrap_or_else(|error| panic!("{error}"));
            assert_eq!(
                qr_outcome.report().termination(),
                DirectSolveTermination::Accepted
            );
            assert!(qr_outcome.report().rank().is_full_rank());
            assert!(qr_outcome.report().residual().is_some());
            let qr_solution = qr_outcome
                .into_solution()
                .unwrap_or_else(|| panic!("full-rank QR solve was rejected"));
            for (&actual, &wanted) in qr_solution.as_slice().iter().zip(qr_expected.as_slice()) {
                assert!((actual - wanted).norm() < tolerance * 4.0);
            }

            let wide = qr_matrix
                .view()
                .adjoint()
                .try_to_matrix()
                .unwrap_or_else(|error| panic!("{error}"));
            let lq = HouseholderDecomposition::factor(
                &wide,
                HouseholderOrientation::Lq,
                scioxide_linalg::HouseholderPivoting::Column,
            )
            .unwrap_or_else(|error| panic!("{error}"));
            assert!(
                lq.analyze_rank(RankPolicy::DimensionScaledEpsilon)
                    .unwrap_or_else(|error| panic!("{error}"))
                    .is_full_rank()
            );
            let reconstructed = lq.reconstruct().unwrap_or_else(|error| panic!("{error}"));
            for (&actual, &wanted) in reconstructed.as_slice().iter().zip(wide.as_slice()) {
                assert!((actual - wanted).norm() < tolerance * 4.0);
            }

            let lower = Matrix::from_vec(
                3,
                3,
                vec![
                    c(3.0, 0.0),
                    c(0.0, 0.0),
                    c(0.0, 0.0),
                    c(0.5, 0.25),
                    c(2.5, 0.0),
                    c(0.0, 0.0),
                    c(-0.25, 0.5),
                    c(0.75, -0.25),
                    c(2.0, 0.0),
                ],
            )
            .unwrap_or_else(|error| panic!("{error}"));
            let cholesky_matrix = lower
                .view()
                .mul_mat(
                    &lower
                        .view()
                        .adjoint()
                        .try_to_matrix()
                        .unwrap_or_else(|error| panic!("{error}")),
                )
                .unwrap_or_else(|error| panic!("{error}"));
            let cholesky = CholeskyDecomposition::factor(
                &cholesky_matrix,
                scioxide_linalg::HermitianInput::Exact,
                scioxide_linalg::CholeskyPivoting::None,
            )
            .unwrap_or_else(|error| panic!("{error}"));
            for (&actual, &wanted) in cholesky.l_matrix().as_slice().iter().zip(lower.as_slice()) {
                assert!((actual - wanted).norm() < tolerance * 4.0);
            }
            let cholesky_rhs = cholesky_matrix
                .view()
                .mul_vec(&expected)
                .unwrap_or_else(|error| panic!("{error}"));
            let cholesky_solution = cholesky
                .solve(&cholesky_rhs)
                .unwrap_or_else(|error| panic!("{error}"));
            for (&actual, &wanted) in cholesky_solution.as_slice().iter().zip(expected.as_slice()) {
                assert!((actual - wanted).norm() < tolerance * 4.0);
            }
        }
    };
}

complex_factorization_suite!(complex32_factorizations, f32, 2e-4);
complex_factorization_suite!(complex64_factorizations, f64, 2e-11);
