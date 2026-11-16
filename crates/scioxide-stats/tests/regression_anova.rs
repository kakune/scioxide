//! Independent identities and published fixtures for regression and ANOVA.
#![allow(clippy::unwrap_used, clippy::expect_used)]
use scioxide_core::{Budget, Complex, ExecutionContext, RealScalar};
use scioxide_linalg::MatrixView;
use scioxide_stats::*;

fn close(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol, "{a} != {b} (tolerance {tol})");
}

fn factor_contrast_covariance<S: RealScalar>() {
    let convert = |x| S::try_from_f64_lossy(x).unwrap();
    let x = [-2.0, -0.5, 0.0, 1.0, 3.0];
    let y = [0.5, 1.2, 0.7, 3.0, 2.5];
    let design: Vec<_> = x.iter().flat_map(|&x| [convert(1.0), convert(x)]).collect();
    let response: Vec<_> = y.iter().copied().map(convert).collect();
    let tolerance = 128.0 * S::EPSILON.into_f64();
    for weights in [[1.0; 5], [1.0, 2.0, 0.5, 0.0, 3.0]] {
        let w: Vec<_> = weights.iter().copied().map(convert).collect();
        let a: f64 = weights.iter().sum();
        let b: f64 = x.iter().zip(weights).map(|(&x, w)| w * x).sum();
        let d: f64 = x.iter().zip(weights).map(|(&x, w)| w * x * x).sum();
        let determinant = a * d - b * b;
        let cy = [1.25, -0.75];
        let z = [
            (d * cy[0] - b * cy[1]) / determinant,
            (a * cy[1] - b * cy[0]) / determinant,
        ];
        let rhs0: f64 = y.iter().zip(weights).map(|(&y, w)| w * y).sum();
        let rhs1: f64 = x
            .iter()
            .zip(y)
            .zip(weights)
            .map(|((&x, y), w)| w * x * y)
            .sum();
        let beta = [
            (d * rhs0 - b * rhs1) / determinant,
            (a * rhs1 - b * rhs0) / determinant,
        ];
        let residuals: Vec<_> = x
            .iter()
            .zip(y)
            .map(|(&x, y)| y - beta[0] - beta[1] * x)
            .collect();
        let degrees = weights.iter().filter(|&&w| w > 0.0).count() as f64 - 2.0;
        let sigma: f64 = residuals
            .iter()
            .zip(weights)
            .map(|(&r, w)| w * r * r)
            .sum::<f64>()
            / degrees;
        for factorization in [RegressionFactorization::Auto, RegressionFactorization::Svd] {
            let model = PreparedRegression::new(
                MatrixView::from_slice(5, 2, &design).unwrap(),
                Weights::<_, Analytic>::new(&w),
                LinearRegressionOptions::default().with_factorization(factorization),
            )
            .unwrap();
            for covariance in [RegressionCovariance::Classical, RegressionCovariance::Hc3] {
                let expected = if covariance == RegressionCovariance::Classical {
                    sigma * (cy[0] * z[0] + cy[1] * z[1])
                } else {
                    x.iter()
                        .zip(&residuals)
                        .zip(weights)
                        .map(|((&x, &r), w)| {
                            let h = w * (d - 2.0 * b * x + a * x * x) / determinant;
                            let v = w * (z[0] + z[1] * x) * r / (1.0 - h);
                            v * v
                        })
                        .sum()
                };
                let result = model
                    .contrast(
                        &response,
                        &[convert(cy[0]), convert(cy[1])],
                        S::ZERO,
                        covariance,
                        Default::default(),
                        Default::default(),
                    )
                    .unwrap();
                close(
                    result.estimate().into_f64(),
                    cy[0] * beta[0] + cy[1] * beta[1],
                    tolerance,
                );
                close(
                    result.standard_error().into_f64(),
                    expected.sqrt(),
                    tolerance,
                );
            }
        }
    }
}

#[test]
fn factor_coordinates_preserve_classical_and_hc3_covariance() {
    factor_contrast_covariance::<f64>();
    factor_contrast_covariance::<f32>();
}

#[test]
fn wide_deficient_inference_and_unit_leverage_follow_selected_factors() {
    for factorization in [RegressionFactorization::Auto, RegressionFactorization::Svd] {
        let options = LinearRegressionOptions::default().with_factorization(factorization);
        let model = PreparedRegression::new(
            MatrixView::from_slice(
                3,
                4,
                &[1.0, 2.0, 0.0, 0.0, 2.0, 4.0, 0.0, 0.0, 3.0, 6.0, 0.0, 0.0],
            )
            .unwrap(),
            Unweighted,
            options,
        )
        .unwrap();
        let fit = model.fit(&[1.0, 3.0, 2.0]).unwrap();
        assert_eq!(fit.rank().unwrap().rank(), 1);
        for covariance in [RegressionCovariance::Classical, RegressionCovariance::Hc3] {
            let result = model
                .contrast(
                    &[1.0, 3.0, 2.0],
                    &[1.0, 2.0, 0.0, 0.0],
                    0.0,
                    covariance,
                    Default::default(),
                    Default::default(),
                )
                .unwrap();
            close(result.estimate(), 13.0 / 14.0, 1e-12);
            assert!(result.standard_error() > 0.0);
            assert!(
                model
                    .contrast(
                        &[1.0, 3.0, 2.0],
                        &[1.0, 0.0, 0.0, 0.0],
                        0.0,
                        covariance,
                        Default::default(),
                        Default::default()
                    )
                    .is_err()
            );
        }
        let saturated_row = PreparedRegression::new(
            MatrixView::from_slice(3, 2, &[1.0, 0.0, 0.0, 1.0, 0.0, 1.0]).unwrap(),
            Unweighted,
            options,
        )
        .unwrap();
        assert!(
            saturated_row
                .contrast(
                    &[2.0, 1.0, 3.0],
                    &[0.0, 1.0],
                    0.0,
                    RegressionCovariance::Hc3,
                    Default::default(),
                    Default::default()
                )
                .is_err()
        );
    }
}

fn extreme_lines<S: RealScalar>(scales: &[S], large: S, tiny: S) {
    let tolerance = if core::mem::size_of::<S>() == 4 {
        3e-6
    } else {
        3e-14
    };
    for &s in scales {
        let x = [s, s + s, s + s + s];
        let fit = linear_regression(&x, &x).unwrap();
        close(fit.line().slope.into_f64(), 1.0, tolerance);
        assert!((fit.line().intercept / s).abs().into_f64() < tolerance);
        for method in [RobustSlopeMethod::TheilSen, RobustSlopeMethod::Siegel] {
            let mut work = RobustSlopeWorkspace::new(3, method, StatisticsOptions::new()).unwrap();
            for intercept in [RobustIntercept::Joint, RobustIntercept::Separate] {
                let robust = work
                    .evaluate(&x, &x, intercept, Default::default())
                    .unwrap();
                close(robust.slope().into_f64(), 1.0, tolerance);
                assert_eq!(robust.intercept(), S::ZERO);
            }
        }
    }
    let half = S::try_from_f64_lossy(0.5).unwrap();
    let x = [-large, S::ZERO, large];
    let y = [-large * half, S::ZERO, large * half];
    let fit = linear_regression(&x, &y).unwrap();
    close(fit.line().slope.into_f64(), 0.5, tolerance);
    assert!((fit.line().intercept / large).abs().into_f64() < tolerance);
    let constant = linear_regression(&x, &[large; 3]).unwrap();
    assert_eq!(constant.line().slope, S::ZERO);
    assert_eq!(constant.line().intercept, large);
    assert!(constant.line().correlation.is_none());
    assert!(linear_regression(&[S::ONE; 3], &y).is_err());
    assert!(linear_regression(&[S::ZERO, tiny], &[S::ZERO, large]).is_err());
    assert!(linear_regression(&[-large, -large * half], &[S::ZERO, large]).is_err());
    let pair = linear_regression(&[S::ZERO, S::ONE], &[S::ONE, S::ONE]).unwrap();
    assert!(
        pair.slope_inference(Default::default(), Default::default())
            .unwrap()
            .is_none()
    );
    assert!(
        pair.intercept_inference(Default::default(), Default::default())
            .unwrap()
            .is_none()
    );
    for method in [RobustSlopeMethod::TheilSen, RobustSlopeMethod::Siegel] {
        let mut work = RobustSlopeWorkspace::new(3, method, StatisticsOptions::new()).unwrap();
        for intercept in [RobustIntercept::Joint, RobustIntercept::Separate] {
            let result = work
                .evaluate(&x, &y, intercept, Default::default())
                .unwrap();
            close(result.slope().into_f64(), 0.5, tolerance);
            assert_eq!(result.intercept(), S::ZERO);
            let constant = work
                .evaluate(&x, &[large; 3], intercept, Default::default())
                .unwrap();
            assert_eq!(constant.slope(), S::ZERO);
            assert_eq!(constant.intercept(), large);
            assert!(
                work.evaluate(&[S::ONE; 3], &y, intercept, Default::default())
                    .is_err()
            );
        }
    }
    for method in [RobustSlopeMethod::TheilSen, RobustSlopeMethod::Siegel] {
        let mut work = RobustSlopeWorkspace::new(2, method, StatisticsOptions::new()).unwrap();
        assert!(
            work.evaluate(
                &[S::ZERO, tiny],
                &[S::ZERO, large],
                RobustIntercept::Joint,
                Default::default()
            )
            .is_err()
        );
        assert!(
            work.evaluate(
                &[-large, -large * half],
                &[S::ZERO, large],
                RobustIntercept::Separate,
                Default::default()
            )
            .is_err()
        );
    }
    // An overflowing individual pair slope is not the selected Siegel median.
    let x = [S::ZERO, tiny, S::ONE, S::ONE + S::ONE];
    let y = [S::ZERO, S::ONE, S::ONE, S::ONE + S::ONE];
    let mut work =
        RobustSlopeWorkspace::new(4, RobustSlopeMethod::Siegel, StatisticsOptions::new()).unwrap();
    for intercept in [RobustIntercept::Joint, RobustIntercept::Separate] {
        let result = work
            .evaluate(&x, &y, intercept, Default::default())
            .unwrap();
        close(result.slope().into_f64(), 1.0, tolerance);
        assert_eq!(result.intercept(), S::ZERO);
    }
}

#[test]
fn line_coefficients_and_robust_medians_retain_extreme_ranges() {
    extreme_lines(&[1e-200_f64, 1e300, f64::from_bits(1)], f64::MAX, 1e-310);
    extreme_lines(&[1e-30_f32, 1e30, f32::from_bits(1)], f32::MAX, 1e-40);
}

#[test]
fn inference_queries_fail_independently_of_coefficients() {
    let fit = linear_regression(&[-1e-310_f64, 0.0, 1e-310], &[1.0, -2.0, 1.0]).unwrap();
    assert!(fit.line().slope.is_finite());
    assert!(
        fit.slope_inference(Default::default(), Default::default())
            .is_err()
    );
    assert!(
        fit.intercept_inference(Default::default(), Default::default())
            .unwrap()
            .is_some()
    );
    let fit = linear_regression(&[-1e-40_f32, 0.0, 1e-40], &[1.0, -2.0, 1.0]).unwrap();
    assert!(
        fit.slope_inference(Default::default(), Default::default())
            .is_err()
    );
    assert!(
        fit.intercept_inference(Default::default(), Default::default())
            .unwrap()
            .is_some()
    );

    let base = 2.0_f64.powi(1000);
    let step = 2.0_f64.powi(950);
    let response = 2.0_f64.powi(1000);
    let fit = linear_regression(
        &[base - step, base, base + step],
        &[response, -response, response],
    )
    .unwrap();
    assert!(
        fit.slope_inference(Default::default(), Default::default())
            .unwrap()
            .is_some()
    );
    assert!(
        fit.intercept_inference(Default::default(), Default::default())
            .is_err()
    );
    let base = 2.0_f32.powi(100);
    let step = 2.0_f32.powi(80);
    let response = 2.0_f32.powi(110);
    let fit = linear_regression(
        &[base - step, base, base + step],
        &[response, -response, response],
    )
    .unwrap();
    assert!(
        fit.slope_inference(Default::default(), Default::default())
            .unwrap()
            .is_some()
    );
    assert!(
        fit.intercept_inference(Default::default(), Default::default())
            .is_err()
    );
}

#[test]
fn intercept_cancellation_precedes_product_projection() {
    for exponent in [1022, 125] {
        let k = 2.0_f64.powi(exponent);
        let fit = linear_regression(&[k, 1.5 * k, 2.0 * k], &[-k, 0.0, k]).unwrap();
        close(fit.line().slope, 2.0, 2e-14);
        close(fit.line().intercept / k, -3.0, 2e-14);
        for method in [RobustSlopeMethod::TheilSen, RobustSlopeMethod::Siegel] {
            let mut work = RobustSlopeWorkspace::new(3, method, StatisticsOptions::new()).unwrap();
            for intercept in [RobustIntercept::Joint, RobustIntercept::Separate] {
                let fit = work
                    .evaluate(
                        &[k, 1.5 * k, 2.0 * k],
                        &[-k, 0.0, k],
                        intercept,
                        Default::default(),
                    )
                    .unwrap();
                close(fit.slope(), 2.0, 2e-14);
                close(fit.intercept() / k, -3.0, 2e-14);
            }
        }
    }
    let k = 2.0_f32.powi(125);
    let x = [k, 1.5 * k, 2.0 * k];
    let y = [-k, 0.0, k];
    let fit = linear_regression(&x, &y).unwrap();
    close(fit.line().slope as f64, 2.0, 2e-6);
    close((fit.line().intercept / k) as f64, -3.0, 2e-6);
    for method in [RobustSlopeMethod::TheilSen, RobustSlopeMethod::Siegel] {
        let mut work = RobustSlopeWorkspace::new(3, method, StatisticsOptions::new()).unwrap();
        for intercept in [RobustIntercept::Joint, RobustIntercept::Separate] {
            let fit = work
                .evaluate(&x, &y, intercept, Default::default())
                .unwrap();
            close(fit.slope() as f64, 2.0, 2e-6);
            close((fit.intercept() / k) as f64, -3.0, 2e-6);
        }
    }
}
fn regression<S: RealScalar>() {
    let values =
        [1., 0., 1., 1., 1., 2., 1., 3., 1., 4.].map(|v| S::try_from_f64_lossy(v).unwrap());
    let y = [1., 2., 4., 4., 5.].map(|v| S::try_from_f64_lossy(v).unwrap());
    let view = MatrixView::from_slice(5, 2, &values).unwrap();
    let model = PreparedRegression::new(view, Unweighted, Default::default()).unwrap();
    let fit = model.fit(&y).unwrap();
    let tol = if core::mem::size_of::<S>() == 4 {
        2e-5
    } else {
        1e-12
    };
    close(
        fit.coefficients()
            .view()
            .get(0)
            .unwrap()
            .try_to_f64_exact()
            .unwrap(),
        1.2,
        tol,
    );
    close(
        fit.coefficients()
            .view()
            .get(1)
            .unwrap()
            .try_to_f64_exact()
            .unwrap(),
        1.,
        tol,
    );
    close(
        fit.weighted_residual_norm()
            .try_to_f64_exact()
            .unwrap()
            .powi(2),
        0.8,
        tol,
    );
    assert_eq!(fit.rank().unwrap().rank(), 2);
    assert_eq!(fit.degrees_of_freedom(), Some(3.));
    let svd = PreparedRegression::new(
        view,
        Unweighted,
        LinearRegressionOptions::default().with_factorization(RegressionFactorization::Svd),
    )
    .unwrap();
    let other = svd.fit(&y).unwrap();
    for (&a, &b) in fit
        .coefficients()
        .as_slice()
        .iter()
        .zip(other.coefficients().as_slice())
    {
        close(
            a.try_to_f64_exact().unwrap(),
            b.try_to_f64_exact().unwrap(),
            tol,
        );
    }
    let contrast = model
        .contrast(
            &y,
            &[S::ZERO, S::ONE],
            S::ZERO,
            RegressionCovariance::Classical,
            ConfidenceProbability::default(),
            TestAlternative::TwoSided,
        )
        .unwrap();
    close(
        contrast.standard_error().try_to_f64_exact().unwrap(),
        (0.8_f64 / 30.).sqrt(),
        tol,
    );
    let hc3 = model
        .contrast(
            &y,
            &[S::ZERO, S::ONE],
            S::ZERO,
            RegressionCovariance::Hc3,
            ConfidenceProbability::default(),
            TestAlternative::TwoSided,
        )
        .unwrap();
    // Independent sandwich sum: sum[(x-xbar)^2 * r^2 / (Sxx^2*(1-h)^2)].
    close(
        hc3.standard_error().try_to_f64_exact().unwrap().powi(2),
        0.02163265306122449,
        tol,
    );
    let mut output = model.output().unwrap();
    let mut workspace = model.workspace().unwrap();
    model
        .fit_into(
            &y,
            &mut output,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    let before = output.coefficients().as_slice().to_vec();
    let mut invalid = y;
    invalid[0] = S::NAN;
    assert!(
        model
            .fit_into(
                &invalid,
                &mut output,
                &mut workspace,
                ExecutionContext::sequential()
            )
            .is_err()
    );
    assert_eq!(output.coefficients().as_slice(), before);
}
#[test]
fn both_precisions() {
    regression::<f32>();
    regression::<f64>();
}
#[test]
fn deficient_minimum_norm_and_estimability() {
    let a = [1., 1., 2., 2., 3., 3.];
    let y = [2., 4., 6.];
    let model = PreparedRegression::new(
        MatrixView::from_slice(3, 2, &a).unwrap(),
        Unweighted,
        Default::default(),
    )
    .unwrap();
    let fit = model.fit(&y).unwrap();
    assert_eq!(fit.rank().unwrap().rank(), 1);
    close(fit.coefficients().view().get(0).unwrap(), 1., 1e-12);
    close(fit.coefficients().view().get(1).unwrap(), 1., 1e-12);
    assert!(
        model
            .contrast(
                &y,
                &[1., 0.],
                0.,
                RegressionCovariance::Classical,
                Default::default(),
                Default::default()
            )
            .is_err()
    );
    assert!(
        model
            .contrast(
                &y,
                &[1., 1.],
                0.,
                RegressionCovariance::Classical,
                Default::default(),
                Default::default()
            )
            .is_ok()
    );
}
#[test]
fn frequency_weights_equal_replication() {
    let a = [1., 0., 1., 1., 1., 3.];
    let y = [1., 3., 3.];
    let w = [2., 1., 3.];
    let weighted = PreparedRegression::new(
        MatrixView::from_slice(3, 2, &a).unwrap(),
        Weights::<_, Frequency>::new(&w),
        Default::default(),
    )
    .unwrap();
    let ar = [1., 0., 1., 0., 1., 1., 1., 3., 1., 3., 1., 3.];
    let yr = [1., 1., 3., 3., 3., 3.];
    let replicated = PreparedRegression::new(
        MatrixView::from_slice(6, 2, &ar).unwrap(),
        Unweighted,
        Default::default(),
    )
    .unwrap();
    let c = weighted
        .contrast(
            &y,
            &[0., 1.],
            0.,
            Default::default(),
            Default::default(),
            Default::default(),
        )
        .unwrap();
    let r = replicated
        .contrast(
            &yr,
            &[0., 1.],
            0.,
            Default::default(),
            Default::default(),
            Default::default(),
        )
        .unwrap();
    close(c.estimate(), r.estimate(), 1e-12);
    close(c.standard_error(), r.standard_error(), 1e-12);
    close(c.p_value(), r.p_value(), 1e-12);
    assert_eq!(c.degrees_of_freedom(), r.degrees_of_freedom());
}
#[test]
fn complex_least_squares() {
    let a = [
        Complex::new(1., 1.),
        Complex::new(2., -1.),
        Complex::new(3., 0.),
    ];
    let beta = Complex::new(2., -3.);
    let y = a.map(|v| v * beta);
    let model = PreparedRegression::new(
        MatrixView::from_slice(3, 1, &a).unwrap(),
        Unweighted,
        Default::default(),
    )
    .unwrap();
    let fit = model.fit(&y).unwrap();
    close(
        (fit.coefficients().view().get(0).unwrap() - beta).norm(),
        0.,
        1e-12,
    );
}
#[test]
fn robust_slopes_and_ties() {
    let x = [0., 1., 2., 3., 4., 5., 6.];
    let y = [1., 3., 5., 7., 9., 11., 100.];
    for method in [RobustSlopeMethod::TheilSen, RobustSlopeMethod::Siegel] {
        let mut work =
            RobustSlopeWorkspace::new(x.len(), method, StatisticsOptions::new()).unwrap();
        let fit = work
            .evaluate(&x, &y, RobustIntercept::Joint, Default::default())
            .unwrap();
        close(fit.slope(), 2., 1e-14);
        close(fit.intercept(), 1., 1e-14);
        assert!(
            work.evaluate(&[1.; 7], &y, RobustIntercept::Joint, Default::default())
                .is_err()
        );
    }
}
#[test]
fn group_statistics_and_two_group_identity() {
    let a = [1., 2., 3.];
    let b = [2., 3., 4.];
    let groups = PreparedGroups::new(&[&a, &b], StatisticsOptions::new()).unwrap();
    let anova = groups.anova(GroupTestMethod::Anova).unwrap();
    close(anova.statistic(), 1.5, 1e-12);
    close(anova.p_value(), 0.2878641347266908, 1e-12);
    let options = PostHocOptions::default().with_evaluations(Budget::try_from(8_000_000).unwrap());
    let pair = groups
        .pairwise_with_options(PostHocMethod::TukeyKramer, options)
        .unwrap();
    close(pair.comparisons()[0].p_value(), anova.p_value(), 1e-6);
    let control = groups
        .compare_control_with_options(0, TestAlternative::TwoSided, options)
        .unwrap();
    close(control.comparisons()[0].p_value(), anova.p_value(), 1e-6);
    close(
        control.comparisons()[0].lower().unwrap(),
        -pair.comparisons()[0].upper().unwrap(),
        5e-6,
    );
}
#[test]
fn published_alexander_govern() {
    let a = [13.75, 13.75, 13.5, 13.5, 13., 13., 13., 12.75, 12.5];
    let b = [14.25, 13., 12.75, 12.5, 12.5, 12.4, 12.3, 11.9, 11.9];
    let c = [14., 14., 13.51, 13.5, 13.5, 13.25, 13., 12.5, 12.5];
    let d = [15., 14., 13.75, 13.59, 13.25, 12.97, 12.5, 12.25, 11.89];
    let result = PreparedGroups::new(&[&a, &b, &c, &d], StatisticsOptions::new())
        .unwrap()
        .alexander_govern()
        .unwrap();
    close(result.statistic(), 4.65087071883494, 1e-11);
    close(result.p_value(), 0.19922132490385214, 1e-11);
}
#[test]
fn rank_test_identities() {
    let k =
        kruskal_wallis_test_with_options(&[&[1., 2., 3.], &[4., 5., 6.]], StatisticsOptions::new())
            .unwrap();
    close(k.statistic(), 27. / 7., 1e-12);
    let f = friedman_test_with_options(
        MatrixView::from_slice(3, 3, &[1., 2., 3., 2., 3., 4., 3., 4., 5.]).unwrap(),
        StatisticsOptions::new(),
    )
    .unwrap();
    close(f.statistic(), 6., 1e-12);
    close(f.p_value(), (-3.0_f64).exp(), 1e-12);
}

#[test]
fn overflowing_group_sums_preserve_anova_and_bartlett() {
    let a = [-1e154_f64, -1e154, 1e154, 1e154];
    let b = [0.0, 0.0, 2e154, 2e154];
    let plan = PreparedGroups::new(&[&a, &b], StatisticsOptions::new()).unwrap();
    let result = plan.anova(GroupTestMethod::Anova).unwrap();
    close(result.statistic(), 1.5, 2e-14);
    assert_eq!(result.denominator_degrees(), Some(6.0));
    let ordinary = PreparedGroups::new(
        &[&[-1., -1., 1., 1.], &[0., 0., 2., 2.]],
        StatisticsOptions::new(),
    )
    .unwrap()
    .anova(GroupTestMethod::Anova)
    .unwrap();
    close(result.p_value(), ordinary.p_value(), 2e-14);
    let bartlett = bartlett_test_with_options(&[&a, &a], ClassicalOptions::default()).unwrap();
    assert_eq!(bartlett.statistic(), 0.0);
    assert_eq!(bartlett.p_value(), 1.0);
}

fn scaled_groups<S: RealScalar>(exponents: &[i32], translation: (i32, i32)) {
    let a = [-1., -1., 1., 1.];
    let b = [0., 0., 2., 2.];
    let c = [-2., 0., 2., 4.];
    let base = PreparedGroups::new(&[&a, &b, &c], StatisticsOptions::new()).unwrap();
    let tolerance = if core::mem::size_of::<S>() == 4 {
        2e-6
    } else {
        5e-13
    };
    let mut frames = exponents
        .iter()
        .map(|&e| (S::ONE.scalbn(e), S::ZERO))
        .collect::<Vec<_>>();
    frames.push((S::ONE.scalbn(translation.1), S::ONE.scalbn(translation.0)));
    for (scale, shift) in frames {
        for sign in [S::ONE, -S::ONE] {
            let transform = |values: [f64; 4]| {
                values.map(|v| shift + sign * scale * S::try_from_f64_lossy(v).unwrap())
            };
            let (x, y, z) = (transform(a), transform(b), transform(c));
            let plan = PreparedGroups::new(&[&x, &y, &z], StatisticsOptions::new()).unwrap();
            for method in [
                GroupTestMethod::Anova,
                GroupTestMethod::Welch,
                GroupTestMethod::AlexanderGovern,
            ] {
                let test = |p: &PreparedGroups<_>| {
                    if method == GroupTestMethod::AlexanderGovern {
                        p.alexander_govern()
                    } else {
                        p.anova(method)
                    }
                };
                let expected = test(&base).unwrap();
                let actual = if method == GroupTestMethod::AlexanderGovern {
                    plan.alexander_govern()
                } else {
                    plan.anova(method)
                }
                .unwrap();
                assert!(
                    (actual.statistic().into_f64() - expected.statistic()).abs() <= tolerance,
                    "{method:?}: {} != {}, scale {}, shift {}, sign {}",
                    actual.statistic().into_f64(),
                    expected.statistic(),
                    scale.into_f64(),
                    shift.into_f64(),
                    sign.into_f64()
                );
                close(actual.p_value().into_f64(), expected.p_value(), tolerance);
                if let Some(df) = expected.denominator_degrees() {
                    close(actual.denominator_degrees().unwrap(), df, tolerance);
                }
            }
        }
    }
}

#[test]
fn group_inference_is_invariant_across_scalar_ranges() {
    scaled_groups::<f64>(&[-1074, -700, 0, 510, 1000], (1000, 950));
    scaled_groups::<f32>(&[-149, -80, 0, 65, 120], (100, 80));
    {
        let a = [-f64::MAX, -f64::MAX, f64::MAX, f64::MAX];
        let b = [-f64::MAX / 2., -f64::MAX / 2., f64::MAX / 2., f64::MAX / 2.];
        let plan = PreparedGroups::new(&[&a, &b], StatisticsOptions::new()).unwrap();
        for method in [GroupTestMethod::Anova, GroupTestMethod::Welch] {
            let result = plan.anova(method).unwrap();
            close(result.statistic(), 0.0, 1e-28);
            close(result.p_value(), 1.0, 1e-14);
        }
        close(plan.alexander_govern().unwrap().statistic(), 0.0, 1e-28);
    }
}

#[test]
fn group_degeneracy_and_final_overflow_are_distinct() {
    use scioxide_core::{NumericalIssue, SciOxideErrorKind, SciOxideErrorView};
    assert!(PreparedGroups::<f64>::new(&[], StatisticsOptions::new()).is_err());
    assert!(PreparedGroups::new(&[&[][..], &[1.]], StatisticsOptions::new()).is_err());
    assert!(PreparedGroups::new(&[&[f64::NAN, 1.], &[1., 2.]], StatisticsOptions::new()).is_err());
    let constant = PreparedGroups::new(&[&[1.; 3], &[2.; 3]], StatisticsOptions::new()).unwrap();
    assert_eq!(
        constant.anova(GroupTestMethod::Anova).unwrap_err().kind(),
        SciOxideErrorKind::InvalidInput
    );
    let mixed = PreparedGroups::new(&[&[0.][..], &[1., 2., 3.]], StatisticsOptions::new()).unwrap();
    assert!(mixed.anova(GroupTestMethod::Anova).is_ok());
    assert!(mixed.anova(GroupTestMethod::Welch).is_err());
    assert!(mixed.alexander_govern().is_err());
    let extreme =
        PreparedGroups::new(&[&[0., 1e-200], &[1., 1.]], StatisticsOptions::new()).unwrap();
    assert!(matches!(
        extreme.anova(GroupTestMethod::Anova).unwrap_err().view(),
        SciOxideErrorView::NumericalInstability {
            issue: NumericalIssue::Overflow,
            ..
        }
    ));
    let heterogeneous = PreparedGroups::new(
        &[&[0., 1e-200], &[1., 1. + f64::EPSILON]],
        StatisticsOptions::new(),
    )
    .unwrap();
    assert!(
        heterogeneous
            .anova(GroupTestMethod::Welch)
            .unwrap()
            .statistic()
            .is_finite()
    );
    assert!(
        heterogeneous
            .alexander_govern()
            .unwrap()
            .statistic()
            .is_finite()
    );
    let extreme =
        PreparedGroups::new(&[&[0., 1e-30_f32], &[1., 1.]], StatisticsOptions::new()).unwrap();
    assert!(matches!(
        extreme.anova(GroupTestMethod::Anova).unwrap_err().view(),
        SciOxideErrorView::NumericalInstability {
            issue: NumericalIssue::Overflow,
            ..
        }
    ));
}
#[test]
fn robust_slopes_select_the_sorted_order_statistics() {
    // Tied ordinary data: every pair slope rounds like binary64 arithmetic, so a
    // fully sorted reference must reproduce the selected medians and Sen ranks.
    let n = 41_usize;
    let x: Vec<f64> = (0..n).map(|i| ((i * 37 + 5) % 23) as f64 * 0.5).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| 2.0 * x[i] + ((i * 7919) % 13) as f64 * 0.25 - 1.0)
        .collect();
    let median = |values: &mut Vec<f64>| {
        values.sort_by(f64::total_cmp);
        let m = values.len();
        if m % 2 == 1 {
            values[m / 2]
        } else {
            (values[m / 2 - 1] + values[m / 2]) / 2.0
        }
    };
    let slope = |i: usize, j: usize| (y[j] - y[i]) / (x[j] - x[i]);
    let mut pairs: Vec<f64> = (0..n)
        .flat_map(|i| (i + 1..n).map(move |j| (i, j)))
        .filter(|&(i, j)| x[i] != x[j])
        .map(|(i, j)| slope(i, j))
        .collect();
    let theil = median(&mut pairs);
    let ties = |v: &[f64]| {
        let mut v = v.to_vec();
        v.sort_by(f64::total_cmp);
        v.chunk_by(|a, b| a == b)
            .map(|run| {
                let t = run.len() as f64;
                t * (t - 1.0) * (2.0 * t + 5.0)
            })
            .sum::<f64>()
    };
    let (count, nf) = (pairs.len() as f64, n as f64);
    let variance = (nf * (nf - 1.0) * (2.0 * nf + 5.0) - ties(&x) - ties(&y)) / 18.0;
    let width = 1.959_963_984_540_054 * variance.sqrt();
    let lo = (((count - width) / 2.0).round() - 1.0).max(0.0) as usize;
    let hi = ((count + width) / 2.0).round() as usize;
    let confidence = ConfidenceProbability::new(0.95).unwrap();
    let mut work =
        RobustSlopeWorkspace::new(n, RobustSlopeMethod::TheilSen, StatisticsOptions::new())
            .unwrap();
    let fit = work
        .evaluate(&x, &y, RobustIntercept::Joint, confidence)
        .unwrap();
    assert_eq!(fit.slope(), theil);
    let interval = fit.slope_interval().unwrap();
    assert_eq!((interval.lower(), interval.upper()), (pairs[lo], pairs[hi]));
    let mut medians: Vec<f64> = (0..n)
        .map(|i| {
            median(
                &mut (0..n)
                    .filter(|&j| x[i] != x[j])
                    .map(|j| slope(i, j))
                    .collect(),
            )
        })
        .collect();
    let siegel = median(&mut medians);
    let mut work =
        RobustSlopeWorkspace::new(n, RobustSlopeMethod::Siegel, StatisticsOptions::new()).unwrap();
    for intercept in [RobustIntercept::Joint, RobustIntercept::Separate] {
        let fit = work.evaluate(&x, &y, intercept, confidence).unwrap();
        assert_eq!(fit.slope(), siegel);
    }
}
#[test]
fn siegel_rows_with_overflowing_differences_replay_in_extended_range() {
    // y[1] - y[0] overflows binary64; point medians 1.5e308, 1e308 and 5e307.
    let (x, y) = ([0., 1., 2.], [-1e308, 1e308, 1e308]);
    let mut work =
        RobustSlopeWorkspace::new(3, RobustSlopeMethod::Siegel, StatisticsOptions::new()).unwrap();
    let fit = work
        .evaluate(&x, &y, RobustIntercept::Joint, Default::default())
        .unwrap();
    close(fit.slope() / 1e308, 1., 1e-15);
    close(fit.intercept() / 1e308, -1., 1e-15);
}
