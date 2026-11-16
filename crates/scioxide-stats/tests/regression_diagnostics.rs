//! Tall OLS/WLS identities and independent exact diagnostic products.
#![allow(clippy::unwrap_used)]
use scioxide_core::{NumericScalar, ProductSum, RealScalar};
use scioxide_linalg::{Matrix, MatrixView};
use scioxide_stats::{
    Analytic, ConfidenceProbability, LinearRegressionOptions, PreparedRegression,
    RegressionCovariance, RegressionFactorization, TestAlternative, Unweighted, Weights,
};

fn exact<T: NumericScalar>(pairs: impl Iterator<Item = (T, T)>) -> T {
    let mut sum = ProductSum::new();
    for (a, b) in pairs {
        sum.add_product(a, b).unwrap();
    }
    sum.finish().unwrap()
}

fn tall<S: RealScalar>() {
    let convert = |x| S::try_from_f64_lossy(x).unwrap();
    let beta = [1.25, -0.5, 2.0, -3.0].map(convert);
    for n in [64, 1024] {
        // Householder QR and SVD solves are normwise backward stable with an
        // O(n ε) bound for n rows; this design has condition number one, so
        // the forward error scales the same way.
        let tolerance = 4.0 * n as f64 * S::EPSILON.into_f64();
        for exponent in [-200, 0, 200] {
            // f32 uses a smaller frame; both cases exercise the guarded and
            // full-range paths with the same independent orthogonal design.
            let exponent = if S::MANTISSA_DIGITS == 24 {
                exponent / 5
            } else {
                exponent
            };
            let scale = S::ONE.scalbn(exponent);
            let values: Vec<_> = (0..n)
                .flat_map(|i| {
                    [
                        S::ONE,
                        if i & 1 == 0 { S::ONE } else { -S::ONE },
                        if i & 2 == 0 { S::ONE } else { -S::ONE },
                        if i & 4 == 0 { S::ONE } else { -S::ONE },
                    ]
                    .map(|x| x * scale)
                })
                .collect();
            let response: Vec<_> = values
                .chunks_exact(4)
                .enumerate()
                .map(|(i, row)| {
                    exact(row.iter().copied().zip(beta))
                        + if (i & 1 == 0) == (i & 2 == 0) {
                            scale / convert(16.0)
                        } else {
                            -scale / convert(16.0)
                        }
                })
                .collect();
            let weights: Vec<_> = (0..n).map(|i| convert(((i / 8) % 3) as f64)).collect();
            for weighted in [false, true] {
                for factorization in [RegressionFactorization::Auto, RegressionFactorization::Svd] {
                    let design = MatrixView::from_slice(n, 4, &values).unwrap();
                    let options =
                        LinearRegressionOptions::default().with_factorization(factorization);
                    let model = if weighted {
                        PreparedRegression::new(
                            design,
                            Weights::<_, Analytic>::new(weights.as_slice()),
                            options,
                        )
                    } else {
                        PreparedRegression::new(design, Unweighted, options)
                    }
                    .unwrap();
                    let fit = model.fit(&response).unwrap();
                    for (&actual, expected) in fit.coefficients().as_slice().iter().zip(beta) {
                        assert!((actual - expected).abs().into_f64() < tolerance);
                    }
                    for (i, row) in values.chunks_exact(4).enumerate() {
                        let expected = exact(
                            row.iter()
                                .copied()
                                .zip(fit.coefficients().as_slice().iter().copied()),
                        );
                        assert_eq!(fit.fitted().as_slice()[i], expected);
                        assert_eq!(fit.residuals().as_slice()[i], response[i] - expected);
                        let root = if weighted { weights[i].sqrt() } else { S::ONE };
                        assert_eq!(
                            fit.weighted_residuals().as_slice()[i],
                            fit.residuals().as_slice()[i] * root
                        );
                    }
                    let residual_squares: f64 = fit
                        .residuals()
                        .as_slice()
                        .iter()
                        .map(|&r| (r / scale).into_f64().powi(2))
                        .sum();
                    let weighted_squares: f64 = fit
                        .weighted_residuals()
                        .as_slice()
                        .iter()
                        .map(|&r| (r / scale).into_f64().powi(2))
                        .sum();
                    assert!(
                        ((fit.residual_norm() / scale).into_f64() - residual_squares.sqrt()).abs()
                            < tolerance
                    );
                    assert!(
                        ((fit.weighted_residual_norm() / scale).into_f64()
                            - weighted_squares.sqrt())
                        .abs()
                            < tolerance
                    );
                    let normal: Vec<_> = (0..4)
                        .map(|j| {
                            exact(
                                values
                                    .chunks_exact(4)
                                    .enumerate()
                                    .zip(fit.weighted_residuals().as_slice())
                                    .map(|((i, row), &r)| {
                                        (
                                            row[j]
                                                * if weighted { weights[i].sqrt() } else { S::ONE },
                                            r,
                                        )
                                    }),
                            )
                        })
                        .collect();
                    let normal = Matrix::from_vec(4, 1, normal).unwrap();
                    let expected = normal
                        .view()
                        .col(0)
                        .unwrap()
                        .norm2_scoped(
                            &Default::default(),
                            scioxide_core::ExecutionContext::sequential(),
                        )
                        .unwrap();
                    if expected == S::ZERO {
                        assert_eq!(fit.normal_residual_norm(), S::ZERO);
                    } else {
                        assert!(
                            (fit.normal_residual_norm() / expected - S::ONE)
                                .abs()
                                .into_f64()
                                < tolerance
                        );
                    }
                    assert_eq!(fit.rank().unwrap().rank(), 4);
                    assert_eq!(
                        fit.degrees_of_freedom(),
                        Some(
                            if weighted {
                                weights.iter().filter(|&&x| x > S::ZERO).count()
                            } else {
                                n
                            } as f64
                                - 4.0
                        )
                    );
                    let degrees = fit.degrees_of_freedom().unwrap();
                    // The noise and each design column are orthogonal on every
                    // eight-row block, including blocks sharing a row weight.
                    let expected_se = 1.0 / (16.0 * degrees.sqrt());
                    let confidence = ConfidenceProbability::default();
                    let critical = confidence
                        .student_t_critical_value_with_options(degrees, Default::default())
                        .unwrap();
                    let inference = model
                        .contrast(
                            &response,
                            &[S::ZERO, S::ONE, S::ZERO, S::ZERO],
                            fit.coefficients().as_slice()[1],
                            RegressionCovariance::Classical,
                            confidence,
                            TestAlternative::TwoSided,
                        )
                        .unwrap();
                    assert!(inference.estimate().abs().into_f64() < tolerance);
                    assert!(
                        (inference.standard_error().into_f64() / expected_se - 1.0).abs()
                            < tolerance
                    );
                    assert!(
                        (inference.p_value().into_f64() - 1.0).abs() < tolerance * 16.0,
                        "n={n}, precision={}, exponent={exponent}: estimate={:?}, se={:?}, p={:?}",
                        S::MANTISSA_DIGITS,
                        inference.estimate(),
                        inference.standard_error(),
                        inference.p_value()
                    );
                    for (actual, expected) in [
                        (inference.interval().lower(), -0.5 - critical * expected_se),
                        (inference.interval().upper(), -0.5 + critical * expected_se),
                    ] {
                        assert!((actual.into_f64() - expected).abs() < tolerance);
                    }
                }
            }
        }
    }
}

#[test]
fn tall_four_column_diagnostics_in_both_precisions_and_factorizations() {
    tall::<f32>();
    tall::<f64>();
}
