#![allow(clippy::unwrap_used, clippy::panic)]
use scioxide_core::AlgebraicScalar;
#[path = "support/fit_solver.rs"]
mod fit_solver;

use scioxide_core::{
    Bounds, Budget, Complex, ExecutionContext, NumericScalar, ProblemToken, RealScalar,
    SolveBudgets, SolveState,
};
use scioxide_linalg::{Matrix, MatrixView, RankPolicy};
use scioxide_stats::*;

#[test]
fn pca_native_rows_and_late_scaled_overflow_publish_atomically() {
    let training = Matrix::from_vec(
        2,
        3,
        vec![-1.0_f64; 3].into_iter().chain([1.0; 3]).collect(),
    )
    .unwrap();
    let pca = PreparedPca::with_options(training.view(), MultivariateOptions::new()).unwrap();
    let data = Matrix::from_vec(
        2,
        3,
        vec![0.0; 3].into_iter().chain([f64::MAX; 3]).collect(),
    )
    .unwrap();
    let mut output = Matrix::from_vec(2, 1, vec![17.0; 2]).unwrap();
    let mut work = pca.workspace(2, 1).unwrap();
    assert!(
        pca.transform_into(
            data.view(),
            PcaScaling::Unwhitened,
            &mut output,
            &mut work,
            ExecutionContext::sequential()
        )
        .is_err()
    );
    assert_eq!(output.as_slice(), &[17.0; 2]);
}

fn close<S: RealScalar>(actual: S, expected: f64, tolerance: f64) {
    let value = actual.try_to_f64_exact().unwrap();
    assert!(
        (value - expected).abs() < tolerance,
        "{value} != {expected}"
    );
}

#[test]
fn linear_odr_rescaling_and_nonidentifiability() {
    let x = Matrix::from_vec(5, 1, vec![-2., -1., 0., 1., 2.]).unwrap();
    let y = [-3., -1., 1., 3., 5.];
    let fit = linear_odr_with_options(
        x.view(),
        &y,
        &[0.5],
        1.,
        scioxide_stats::OdrOptions::new().with_max_bytes(usize::MAX),
    )
    .unwrap();
    close(fit.coefficients()[0], 1., 1e-12);
    close(fit.coefficients()[1], 2., 1e-12);
    close(fit.report().chi_square().unwrap(), 0., 1e-12);
    assert!(
        linear_odr_with_options(
            x.view(),
            &y,
            &[0.],
            1.,
            scioxide_stats::OdrOptions::new().with_max_bytes(usize::MAX)
        )
        .is_err()
    );
    let noisy = [-3.2, -0.5, 1.1, 3.1, 4.5];
    let noisy_fit = linear_odr_with_options(
        x.view(),
        &noisy,
        &[1.],
        1.,
        scioxide_stats::OdrOptions::new().with_max_bytes(usize::MAX),
    )
    .unwrap();
    let x_mean = 0.0;
    let y_mean = noisy.iter().sum::<f64>() / 5.;
    let sxx = x
        .as_slice()
        .iter()
        .map(|v| (v - x_mean) * (v - x_mean))
        .sum::<f64>();
    let syy = noisy
        .iter()
        .map(|v| (v - y_mean) * (v - y_mean))
        .sum::<f64>();
    let sxy = x
        .as_slice()
        .iter()
        .zip(noisy)
        .map(|(a, b)| (a - x_mean) * (b - y_mean))
        .sum::<f64>();
    let expected = (syy - sxx + ((syy - sxx) * (syy - sxx) + 4. * sxy * sxy).sqrt()) / (2. * sxy);
    close(noisy_fit.coefficients()[1], expected, 1e-11);
    close(noisy_fit.coefficients()[0], y_mean, 1e-11);
    let scaled = Matrix::from_vec(5, 1, x.as_slice().iter().map(|v| v * 10.).collect()).unwrap();
    let scaled_fit = linear_odr_with_options(
        scaled.view(),
        &noisy,
        &[10.],
        1.,
        scioxide_stats::OdrOptions::new().with_max_bytes(usize::MAX),
    )
    .unwrap();
    close(scaled_fit.coefficients()[1], expected / 10., 1e-11);
}

#[test]
fn injected_grid_fits_noisy_measurement_error_data() {
    let x = [-0.5, 0.2, 1.5];
    let y = [-1., 1., 3.];
    let sx = [0.5; 3];
    let sy = [0.2; 3];
    let model = AffineOdrModel::new(1).unwrap();
    let odr = PreparedOdr::new(
        &model,
        MatrixView::from_slice(3, 1, &x).unwrap(),
        MatrixView::from_slice(3, 1, &y).unwrap(),
        MatrixView::from_slice(3, 1, &sx).unwrap(),
        MatrixView::from_slice(3, 1, &sy).unwrap(),
        scioxide_stats::OdrOptions::new(),
    )
    .unwrap();
    let initial = odr.initial_parameters(&[1., 1.]).unwrap();
    let initial_value = odr.value(&initial, &mut ()).unwrap();
    let bounds = Bounds::new(vec![
        (Some(0.), Some(2.)),
        (Some(1.), Some(3.)),
        (Some(-0.5), Some(0.5)),
        (Some(-0.5), Some(0.5)),
        (Some(-0.5), Some(0.5)),
    ])
    .unwrap();
    let integral = [false; 5];
    let mut output = vec![9.; 5];
    let mut workspace = FitWorkspace::new(
        5,
        scioxide_stats::StatisticsOptions::new().with_max_bytes(1024),
    )
    .unwrap();
    let mut solver = fit_solver::GridSolver { intervals: 4 };
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut context = state
        .begin_solve(
            ExecutionContext::sequential(),
            scioxide_core::SolveIdentity::new(
                ProblemToken::new(2),
                scioxide_core::algorithm_id!("stats", "odr-grid", 1),
                0,
                0,
            ),
            &mut budgets,
        )
        .unwrap();
    let report = scioxide_stats::fit_into(
        FitRequest {
            objective: &odr,
            derivatives: Some(&odr),
            bounds: &bounds,
            integrality: &integral,
            initial: &initial,
        },
        &mut solver,
        &mut output,
        &mut workspace,
        &mut context,
    )
    .unwrap();
    assert_eq!(report.termination(), StatsTermination::Converged);
    assert!(report.objective().unwrap() < initial_value);
    assert!((output[1] - 2.).abs() < 0.6);
}

#[test]
fn huber_scatter_handles_outlier_and_budget() {
    let data = Matrix::from_vec(
        8,
        2,
        vec![
            -2., -1., -1., 1., 0., -1., 1., 1., 2., -1., 3., 1., 4., -1., 30., 40.,
        ],
    )
    .unwrap();
    let (mean, _) =
        sample_covariance_with_options(data.view(), MultivariateOptions::new()).unwrap();
    let options = HuberScatterOptions::default()
        .with_iterations(Budget::try_from(60).unwrap())
        .with_tuning(2.)
        .with_tolerance(1e-8)
        .with_max_bytes(usize::MAX);
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut context = state
        .begin_solve(
            ExecutionContext::sequential(),
            scioxide_core::SolveIdentity::new(
                ProblemToken::new(0),
                scioxide_core::algorithm_id!("stats-test", "scatter-parent", 1),
                0,
                0,
            ),
            &mut budgets,
        )
        .unwrap();
    let fit = PreparedHuberScatter::new(data.view(), options)
        .and_then(|plan| plan.fit_scoped(&mut context))
        .unwrap();
    assert!(fit.location()[0] < mean[0]);
    assert!(fit.scatter().view().get(0, 0).unwrap() > 0.);
    assert!(fit.iterations() > 0);
    assert!(
        PreparedHuberScatter::new(data.view(), options.with_max_bytes(1))
            .and_then(|plan| plan.fit_scoped(&mut context))
            .is_err()
    );
    drop(context);
    let mut budget_state = SolveState::new();
    let mut budget = SolveBudgets::unbounded();
    let mut budget_context = budget_state
        .begin_solve(
            ExecutionContext::sequential(),
            scioxide_core::SolveIdentity::new(
                ProblemToken::new(0),
                scioxide_core::algorithm_id!("stats-test", "scatter-parent", 1),
                0,
                0,
            ),
            &mut budget,
        )
        .unwrap();
    let partial = PreparedHuberScatter::new(
        data.view(),
        options
            .with_iterations(Budget::try_from(1).unwrap())
            .with_tolerance(1e-30),
    )
    .and_then(|plan| plan.fit_scoped(&mut budget_context))
    .unwrap();
    assert_eq!(partial.iterations(), 1);
    assert_eq!(partial.termination(), StatsTermination::BudgetExhausted);
}

#[test]
fn strided_views_weighted_replication_and_resource_limits() {
    let backing = [1.0, 99.0, 2.0, 99.0, 3.0, 99.0];
    let view = MatrixView::from_slice_with_strides(3, 1, &backing, 2, 1).unwrap();
    let (mean, cov) = frequency_covariance_with_options(
        view,
        Weights::<_, Frequency>::new(&[2u64, 1, 1]),
        MultivariateOptions::new().with_max_bytes(1024),
    )
    .unwrap();
    close(mean[0], 1.75, 1e-12);
    close(cov.view().get(0, 0).unwrap(), 11. / 12., 1e-12);
    assert!(
        sample_covariance_with_options(view, MultivariateOptions::new().with_max_bytes(1)).is_err()
    );
    assert!(
        PreparedPca::with_options(
            view,
            scioxide_stats::MultivariateOptions::new()
                .with_rank_policy(RankPolicy::DimensionScaledEpsilon)
                .with_max_bytes(1)
        )
        .is_err()
    );
    assert!(
        linear_odr_with_options(
            view,
            &[1., 2., 3.],
            &[1.],
            1.,
            scioxide_stats::OdrOptions::new().with_max_bytes(1)
        )
        .is_err()
    );
    let model = AffineOdrModel::new(1).unwrap();
    let y = MatrixView::from_slice(3, 1, &[1., 2., 3.]).unwrap();
    let sigma = MatrixView::from_slice(3, 1, &[1., 1., 1.]).unwrap();
    let odr = PreparedOdr::new(
        &model,
        view,
        y,
        sigma,
        sigma,
        scioxide_stats::OdrOptions::new().with_max_bytes(1),
    )
    .unwrap();
    assert!(odr.initial_parameters(&[0., 1.]).is_err());
}

#[test]
fn covariance_normalizes_before_final_rounding() {
    let high = 0.75 * f64::MAX;
    let equal = Matrix::from_vec(2, 1, vec![high, high]).unwrap();
    let (mean, cov) =
        sample_covariance_with_options(equal.view(), MultivariateOptions::new()).unwrap();
    assert!((mean[0] - high).abs() / high < 1e-15);
    assert_eq!(cov.view().get(0, 0).unwrap(), 0.);
    let spread = Matrix::from_vec(3, 1, vec![1e154, -1e154, 0.]).unwrap();
    let (_, cov) =
        sample_covariance_with_options(spread.view(), MultivariateOptions::new()).unwrap();
    assert!((cov.view().get(0, 0).unwrap() / 1e308 - 1.).abs() < 1e-14);
}

#[test]
fn complex_cca_and_f32_odr() {
    let real = Matrix::from_vec(4, 2, vec![1., 0., 0., 1., -1., 0., 0., -1.]).unwrap();
    let complex = Matrix::from_vec(
        4,
        2,
        real.as_slice()
            .iter()
            .map(|&v| Complex::new(v, v * 0.25))
            .collect(),
    )
    .unwrap();
    let cca = PreparedCca::with_options(
        complex.view(),
        complex.view(),
        scioxide_stats::MultivariateOptions::new()
            .with_rank_policy(RankPolicy::DimensionScaledEpsilon)
            .with_max_bytes(usize::MAX),
    )
    .unwrap();
    assert_eq!(cca.correlations().len(), 2);
    for &r in cca.correlations() {
        close(r, 1., 1e-10);
    }
    let x = [0.0_f32, 1., 2.];
    let y = [1.0_f32, 3., 5.];
    let sx = [1.0_f32; 3];
    let sy = [1.0_f32; 3];
    let model = AffineOdrModel::new(1).unwrap();
    let odr = PreparedOdr::new(
        &model,
        MatrixView::from_slice(3, 1, &x).unwrap(),
        MatrixView::from_slice(3, 1, &y).unwrap(),
        MatrixView::from_slice(3, 1, &sx).unwrap(),
        MatrixView::from_slice(3, 1, &sy).unwrap(),
        scioxide_stats::OdrOptions::new(),
    )
    .unwrap();
    let z = odr.initial_parameters(&[1.0_f32, 2.]).unwrap();
    close(odr.value(&z, &mut ()).unwrap(), 0., 1e-6);
}

fn scalar<T: NumericScalar>(real: f64, imaginary: f64) -> T {
    T::from_components(
        T::Real::try_from_f64_lossy(real).unwrap(),
        T::Real::try_from_f64_lossy(imaginary).unwrap(),
    )
}

fn relative<S: RealScalar>(actual: S, expected: f64, tolerance: f64) {
    let actual = actual.try_to_f64_exact().unwrap();
    assert!(actual.is_finite());
    assert!(
        (actual / expected - 1.).abs() <= tolerance,
        "{actual} != {expected} (relative tolerance {tolerance})"
    );
}

fn exact_frequency_limits<S: RealScalar + NumericScalar<Real = S>>() {
    let limit = 2_f64.powi(S::MANTISSA_DIGITS as i32);
    let s = |v| S::try_from_f64_lossy(v).unwrap();
    let data = Matrix::from_vec(2, 1, vec![s(0.), s(1.)]).unwrap();
    for count in [limit - 1., limit] {
        let (_, covariance) = frequency_covariance_with_options(
            data.view(),
            Weights::<_, Frequency>::new(&[(count as u64) - 1, 1]),
            MultivariateOptions::new(),
        )
        .unwrap();
        // A replicated Bernoulli sample with one success has sample variance 1/n.
        relative(
            covariance.as_slice()[0],
            1. / count,
            4. * S::EPSILON.try_to_f64_exact().unwrap(),
        );
    }
    let max_count = 1u64 << S::MANTISSA_DIGITS;
    for counts in [
        [max_count, 1],
        [1, max_count],
        [max_count, max_count],
        [max_count * 2, 0],
        [0, 0],
        [1, 0],
        [u64::MAX, 1],
    ] {
        assert!(
            frequency_covariance_with_options(
                data.view(),
                Weights::<_, Frequency>::new(&counts),
                MultivariateOptions::new()
            )
            .is_err()
        );
    }
    let finite_zero = Matrix::from_vec(
        2,
        1,
        vec![s(f64::MAX.min(S::MAX.try_to_f64_exact().unwrap())), s(-1.)],
    )
    .unwrap();
    let (_, covariance) = frequency_covariance_with_options(
        finite_zero.view(),
        Weights::<_, Frequency>::new(&[0u64, 2]),
        MultivariateOptions::new(),
    )
    .unwrap();
    assert_eq!(covariance.as_slice()[0], S::ZERO);
    let nonfinite_zero = Matrix::from_vec(2, 1, vec![S::NAN, s(1.)]).unwrap();
    assert!(
        frequency_covariance_with_options(
            nonfinite_zero.view(),
            Weights::<_, Frequency>::new(&[0u64, 2]),
            MultivariateOptions::new()
        )
        .is_err()
    );
}

#[test]
fn frequency_totals_are_exact_and_zero_weights_preserve_validation() {
    exact_frequency_limits::<f32>();
    exact_frequency_limits::<f64>();
}

fn shrinkage_scales<T: NumericScalar>(scales: &[f64], tolerance: f64) {
    for &scale in scales {
        let data = Matrix::from_vec(
            4,
            2,
            [-2., -1., 1., 2.]
                .into_iter()
                .flat_map(|v| [scalar::<T>(v * scale, v * scale); 2])
                .collect(),
        )
        .unwrap();
        let (alpha, covariance) =
            ledoit_wolf_covariance_with_options(data.view(), MultivariateOptions::new()).unwrap();
        // MLE entries = 5/2, delta = 25/2, beta = 36:
        // beta / (n² delta) = 9/50. Complex (1+i) scales every entry by two.
        relative(alpha, 9. / 50., tolerance);
        let sample = (if T::IS_REAL { 10. / 3. } else { 20. / 3. }) * scale * scale;
        for (i, &entry) in covariance.as_slice().iter().enumerate() {
            assert!(entry.has_finite_components());
            relative(
                entry.real_component(),
                if i == 0 || i == 3 {
                    sample
                } else {
                    sample * 41. / 50.
                },
                tolerance,
            );
            close(entry.imaginary_component(), 0., tolerance);
        }
        assert_eq!(covariance.as_slice()[1], covariance.as_slice()[2].conj());
        for intensity in [0., 0.5, 1.] {
            let intensity = T::Real::try_from_f64_lossy(intensity).unwrap();
            let shrunk = shrink_covariance(covariance.view(), intensity).unwrap();
            assert!(shrunk.as_slice().iter().all(|v| v.has_finite_components()));
            assert_eq!(shrunk.as_slice()[1], shrunk.as_slice()[2].conj());
        }
        let zero = Matrix::from_vec(3, 2, vec![scalar::<T>(scale, -scale); 6]).unwrap();
        let (alpha, covariance) =
            ledoit_wolf_covariance_with_options(zero.view(), MultivariateOptions::new()).unwrap();
        assert_eq!(alpha, T::Real::ONE);
        assert!(covariance.as_slice().iter().all(|v| *v == T::ZERO));
        let isotropic = Matrix::from_vec(
            4,
            2,
            vec![
                scalar::<T>(scale, scale),
                T::ZERO,
                scalar::<T>(-scale, -scale),
                T::ZERO,
                T::ZERO,
                scalar::<T>(scale, scale),
                T::ZERO,
                scalar::<T>(-scale, -scale),
            ],
        )
        .unwrap();
        let (alpha, covariance) =
            ledoit_wolf_covariance_with_options(isotropic.view(), MultivariateOptions::new())
                .unwrap();
        assert_eq!(alpha, T::Real::ONE);
        assert_eq!(covariance.as_slice()[1], T::ZERO);
        assert_eq!(covariance.as_slice()[0], covariance.as_slice()[3]);
    }
}

#[test]
fn ledoit_wolf_intensity_is_invariant_under_extreme_unit_changes() {
    shrinkage_scales::<f64>(&[1., 1e100, 1e150, 1e-100, 1e-150], 4e-14);
    shrinkage_scales::<Complex<f64>>(&[1., 1e100, 1e150, 1e-100, 1e-150], 4e-14);
    shrinkage_scales::<f32>(&[1., 1e10, 1e15, 1e-10, 1e-15], 2e-6);
    shrinkage_scales::<Complex<f32>>(&[1., 1e10, 1e15, 1e-10, 1e-15], 2e-6);
}

fn range_safe_derived<T: NumericScalar>(scale: f64, tolerance: f64) {
    let maximum = T::Real::MAX.try_to_f64_exact().unwrap();
    let covariance = Matrix::from_vec(
        2,
        2,
        vec![
            scalar::<T>(maximum, 0.),
            T::ZERO,
            T::ZERO,
            scalar::<T>(maximum, 0.),
        ],
    )
    .unwrap();
    for intensity in [0., 0.5, 1.] {
        let shrunk = shrink_covariance(
            covariance.view(),
            T::Real::try_from_f64_lossy(intensity).unwrap(),
        )
        .unwrap();
        assert_eq!(shrunk, covariance);
    }
    let data = Matrix::from_vec(
        4,
        1,
        vec![
            scalar::<T>(-scale, 0.),
            scalar::<T>(-scale, 0.),
            scalar::<T>(scale, 0.),
            scalar::<T>(scale, 0.),
        ],
    )
    .unwrap();
    let pca = PreparedPca::with_options(
        data.view(),
        scioxide_stats::MultivariateOptions::new()
            .with_rank_policy(RankPolicy::DimensionScaledEpsilon)
            .with_max_bytes(usize::MAX),
    )
    .unwrap();
    assert_eq!(pca.rank().rank(), 1);
    relative(pca.variances()[0], scale * (scale * (4. / 3.)), tolerance);
    let overflow = Matrix::from_vec(
        2,
        1,
        vec![scalar::<T>(-scale * 2., 0.), scalar::<T>(scale * 2., 0.)],
    )
    .unwrap();
    for result in [
        sample_covariance_with_options(overflow.view(), MultivariateOptions::new()).map(|_| ()),
        frequency_covariance_with_options(
            overflow.view(),
            Weights::<_, Frequency>::new(&[1u64; 2]),
            MultivariateOptions::new(),
        )
        .map(|_| ()),
        ledoit_wolf_covariance_with_options(overflow.view(), MultivariateOptions::new())
            .map(|_| ()),
        PreparedPca::with_options(
            overflow.view(),
            scioxide_stats::MultivariateOptions::new()
                .with_rank_policy(RankPolicy::DimensionScaledEpsilon)
                .with_max_bytes(usize::MAX),
        )
        .map(|_| ()),
    ] {
        assert_eq!(
            result.unwrap_err().kind(),
            scioxide_core::SciOxideErrorKind::NumericalInstability
        );
    }
}

#[test]
fn trace_averages_and_pca_variances_apply_normalization_before_projection() {
    range_safe_derived::<f64>(1e154, 2e-14);
    range_safe_derived::<Complex<f64>>(1e154, 2e-14);
    range_safe_derived::<f32>(1e19, 2e-6);
    range_safe_derived::<Complex<f32>>(1e19, 2e-6);
}

fn complex_shrinkage_phase<S: RealScalar>(scale: f64, tolerance: f64) {
    let data = Matrix::from_vec(
        4,
        2,
        [-2., -1., 1., 2.]
            .into_iter()
            .flat_map(|v| {
                let z = scalar::<Complex<S>>(v * scale, v * scale);
                [z, z.conj()]
            })
            .collect(),
    )
    .unwrap();
    let (alpha, covariance) =
        ledoit_wolf_covariance_with_options(data.view(), MultivariateOptions::new()).unwrap();
    relative(alpha, 0.18, tolerance);
    let off = covariance.as_slice()[1];
    assert_eq!(off.re, S::ZERO);
    relative(off.im, -(20. / 3.) * scale * scale * 0.82, tolerance);
    assert_eq!(covariance.as_slice()[2], off.conj());
    let diagonal = covariance.as_slice()[0].re;
    assert_eq!(diagonal, covariance.as_slice()[3].re);
    for fraction in [0., 0.25, 1.] {
        let intensity = S::try_from_f64_lossy(fraction).unwrap();
        let shrunk = shrink_covariance(covariance.view(), intensity).unwrap();
        assert_eq!(shrunk.as_slice()[1], shrunk.as_slice()[2].conj());
        assert_eq!(shrunk.as_slice()[0].re, diagonal);
        if fraction == 1. {
            assert_eq!(shrunk.as_slice()[1], Complex::new(S::ZERO, S::ZERO));
        } else {
            relative(
                shrunk.as_slice()[1].im,
                off.im.try_to_f64_exact().unwrap() * (1. - fraction),
                tolerance,
            );
        }
    }
}

#[test]
fn complex_shrinkage_preserves_off_diagonal_phase_at_extreme_scales() {
    for scale in [1., 1e150, 1e-150] {
        complex_shrinkage_phase::<f64>(scale, 4e-14);
    }
    for scale in [1., 1e15, 1e-15] {
        complex_shrinkage_phase::<f32>(scale, 2e-6);
    }
}

#[test]
fn shrinkage_endpoints_validate_inputs_and_constant_outer_products_select_zero() {
    let data = Matrix::from_vec(2, 2, vec![-1., -1., 1., 1.]).unwrap();
    let (alpha, covariance) =
        ledoit_wolf_covariance_with_options(data.view(), MultivariateOptions::new()).unwrap();
    assert_eq!(alpha, 0.);
    assert_eq!(covariance.as_slice(), &[2.; 4]);
    for intensity in [0., 1.] {
        for invalid in [
            vec![-1., 0., 0., 1.],
            vec![1., 2., 1., 1.],
            vec![1., f64::NAN, f64::NAN, 1.],
            vec![f64::INFINITY, 0., 0., 1.],
        ] {
            assert!(
                shrink_covariance(MatrixView::from_slice(2, 2, &invalid).unwrap(), intensity)
                    .is_err()
            );
        }
    }
}

#[test]
fn precise_transforms_preserve_outputs_on_late_numerical_failure() {
    let base = 2_f64.powi(53);
    let training = Matrix::from_vec(2, 1, vec![base, base + 2.]).unwrap();
    let pca = PreparedPca::with_options(
        training.view(),
        scioxide_stats::MultivariateOptions::new()
            .with_rank_policy(RankPolicy::DimensionScaledEpsilon)
            .with_max_bytes(usize::MAX),
    )
    .unwrap();
    let invalid = Matrix::from_vec(2, 1, vec![base, f64::NAN]).unwrap();
    let mut output = Matrix::from_vec(2, 1, vec![9.; 2]).unwrap();
    let prior = output.clone();
    let mut work = PcaWorkspace::workspace_requirements(
        2,
        1,
        scioxide_stats::MultivariateOptions::new().with_max_bytes(usize::MAX),
    )
    .and_then(|req| PcaWorkspace::workspace(&req))
    .unwrap();
    assert!(
        pca.transform_into(
            invalid.view(),
            scioxide_stats::PcaScaling::Unwhitened,
            &mut output,
            &mut work,
            scioxide_core::ExecutionContext::sequential()
        )
        .is_err()
    );
    assert_eq!(output, prior);
    let overflowing_scores = Matrix::from_vec(2, 1, vec![0., f64::MAX]).unwrap();
    let error = pca
        .reconstruct_into(
            overflowing_scores.view(),
            scioxide_stats::PcaScaling::Whitened,
            &mut output,
            &mut work,
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap_err();
    assert_eq!(
        error.kind(),
        scioxide_core::SciOxideErrorKind::NumericalInstability
    );
    assert_eq!(output, prior);
    pca.transform_into(
        training.view(),
        scioxide_stats::PcaScaling::Unwhitened,
        &mut output,
        &mut work,
        scioxide_core::ExecutionContext::sequential(),
    )
    .unwrap();

    let cca = PreparedCca::with_options(
        training.view(),
        training.view(),
        scioxide_stats::MultivariateOptions::new()
            .with_rank_policy(RankPolicy::DimensionScaledEpsilon)
            .with_max_bytes(usize::MAX),
    )
    .unwrap();
    let mut xs = prior.clone();
    let mut ys = prior.clone();
    let mut work = CcaWorkspace::workspace_requirements(
        2,
        1,
        scioxide_stats::MultivariateOptions::new().with_max_bytes(usize::MAX),
    )
    .and_then(|req| CcaWorkspace::workspace(&req))
    .unwrap();
    assert!(
        cca.scores_into(
            training.view(),
            invalid.view(),
            &mut xs,
            &mut ys,
            &mut work,
            scioxide_core::ExecutionContext::sequential()
        )
        .is_err()
    );
    assert_eq!(xs, prior);
    assert_eq!(ys, prior);
    cca.scores_into(
        training.view(),
        training.view(),
        &mut xs,
        &mut ys,
        &mut work,
        scioxide_core::ExecutionContext::sequential(),
    )
    .unwrap();
}

#[test]
fn preparation_constant_rank_behavior_is_preserved() {
    let data = Matrix::<f64>::zeros(4, 3).unwrap();
    let policy = RankPolicy::DimensionScaledEpsilon;
    assert_eq!(
        PreparedPca::with_options(
            data.view(),
            scioxide_stats::MultivariateOptions::new()
                .with_rank_policy(policy)
                .with_max_bytes(usize::MAX)
        )
        .unwrap()
        .rank()
        .rank(),
        0
    );
    assert!(
        PreparedCca::with_options(
            data.view(),
            data.view(),
            scioxide_stats::MultivariateOptions::new()
                .with_rank_policy(policy)
                .with_max_bytes(usize::MAX)
        )
        .is_err()
    );
}

fn whitening_ignores_reported_variance_rounding<S: RealScalar>(magnitude: f64) {
    let a = S::try_from_f64_lossy(magnitude).unwrap();
    let data = [-a, a];
    let view = MatrixView::from_slice(2, 1, &data).unwrap();
    let pca = PreparedPca::with_options(view, MultivariateOptions::new()).unwrap();
    // The exact variance 2a² may round, underflow or become subnormal in S.
    let whitened = pca.transform(view, PcaScaling::Whitened).unwrap();
    let half = core::f64::consts::FRAC_1_SQRT_2;
    let tolerance = 4.0 * S::EPSILON.into_f64();
    let orientation = pca.components().as_slice()[0].into_f64().signum();
    for (value, sign) in whitened.as_slice().iter().zip([-1.0, 1.0]) {
        let value = value.into_f64() * orientation;
        assert!((value - sign * half).abs() <= tolerance, "{value}");
    }
    let restored = pca
        .reconstruct(whitened.view(), PcaScaling::Whitened)
        .unwrap();
    for (&restored, &original) in restored.as_slice().iter().zip(&data) {
        assert!((restored - original).abs().into_f64() <= tolerance * magnitude);
    }
}

#[test]
fn pca_whitening_uses_the_unrounded_variance() {
    // f32: the variance 2a² is subnormal for a = 2e-23 and underflows for 1e-23.
    whitening_ignores_reported_variance_rounding::<f32>(2e-23);
    whitening_ignores_reported_variance_rounding::<f32>(1e-23);
    // f64: 8e-324 rounds to the smallest subnormal, a 10% variance error.
    whitening_ignores_reported_variance_rounding::<f64>(2e-162);
    whitening_ignores_reported_variance_rounding::<f64>(1.0);
    let x = [-1e-23_f32, 1e-23];
    let pca = PreparedPca::with_options(
        MatrixView::from_slice(2, 1, &x).unwrap(),
        MultivariateOptions::new(),
    )
    .unwrap();
    assert_eq!(pca.variances(), [0.0]);
    assert_eq!(pca.rank().rank(), 1);
}

#[test]
fn undersized_pca_and_cca_workspaces_report_their_shortfall() {
    use scioxide_core::{Capacity, SciOxideErrorView};
    let data = [1.0_f64, 2., 2., 1., 3., 5., 4., 3.];
    let view = MatrixView::from_slice(4, 2, &data).unwrap();
    let context = ExecutionContext::sequential();
    let shortfall = |error: scioxide_core::SciOxideError| {
        let SciOxideErrorView::ResourceLimit {
            resource,
            required,
            limit,
        } = error.view()
        else {
            panic!("expected a workspace shortfall, got {error}");
        };
        assert!(required > limit, "{error}");
        (resource.as_str(), required, limit)
    };
    let pca = PreparedPca::with_options(view, MultivariateOptions::new()).unwrap();
    let mut small = pca.workspace(1, 2).unwrap();
    let req = pca.workspace_requirements(4, 2).unwrap();
    assert_eq!(small.check(&req).unwrap(), Capacity::NeedsGrowth);
    let mut scores = Matrix::zeros(4, 2).unwrap();
    let error = pca
        .transform_into(
            view,
            PcaScaling::Unwhitened,
            &mut scores,
            &mut small,
            context,
        )
        .err()
        .unwrap();
    assert_eq!(shortfall(error), ("pca_workspace_elements", 8, 2));
    small.try_reserve(&req).unwrap();
    assert_eq!(small.check(&req).unwrap(), Capacity::Sufficient);
    pca.transform_into(
        view,
        PcaScaling::Unwhitened,
        &mut scores,
        &mut small,
        context,
    )
    .unwrap();

    let cca = PreparedCca::with_options(view, view, MultivariateOptions::new()).unwrap();
    let r = cca.correlations().len();
    let mut paired = cca.workspace(1).unwrap();
    let mut xs = Matrix::zeros(4, r).unwrap();
    let mut ys = Matrix::zeros(4, r).unwrap();
    let error = cca
        .scores_into(view, view, &mut xs, &mut ys, &mut paired, context)
        .err()
        .unwrap();
    assert_eq!(shortfall(error), ("cca_workspace_elements", 4 * r, r));
    paired
        .try_reserve(&cca.workspace_requirements(4).unwrap())
        .unwrap();
    cca.scores_into(view, view, &mut xs, &mut ys, &mut paired, context)
        .unwrap();
}

#[test]
fn covariance_budgets_come_from_multivariate_options() {
    let data = [1.0_f64, 2., 2., 1., 3., 5., 4., 3.];
    let view = MatrixView::from_slice(4, 2, &data).unwrap();
    let options = MultivariateOptions::new().with_max_bytes(1);
    for error in [
        sample_covariance_with_options(view, options).err(),
        ledoit_wolf_covariance_with_options(view, options)
            .map(|_| ())
            .err(),
        frequency_covariance_with_options(view, Weights::<_, Frequency>::new(&[1u64; 4]), options)
            .map(|_| ())
            .err(),
    ] {
        assert_eq!(
            error.unwrap().kind(),
            scioxide_core::SciOxideErrorKind::ResourceLimit
        );
    }
}

#[test]
fn affine_odr_model_rejects_zero_predictors() {
    assert_eq!(
        AffineOdrModel::new(0).err().unwrap().kind(),
        scioxide_core::SciOxideErrorKind::InvalidInput
    );
    assert_eq!(
        AffineOdrModel::new(usize::MAX).err().unwrap().kind(),
        scioxide_core::SciOxideErrorKind::SizeOverflow
    );
    assert!(AffineOdrModel::new(2).is_ok());
}
