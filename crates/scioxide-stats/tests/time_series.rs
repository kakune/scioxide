#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! Lag summaries, spectra and ARIMA fitting through the public API.
use scioxide_core::{Complex, ExecutionContext, NumericScalar, RealScalar};
use scioxide_stats::*;
#[test]
fn summary_normalizations_masks_and_transactions() {
    for statistic in [
        TimeSeriesStatistic::Covariance,
        TimeSeriesStatistic::Acf,
        TimeSeriesStatistic::Pacf,
    ] {
        let result = time_series_summary_with_options(
            &[1., 2., 3.],
            None,
            statistic,
            2,
            TimeSeriesOptions::new(),
            StatisticsOptions::new(),
        )
        .unwrap();
        let expected = if statistic == TimeSeriesStatistic::Covariance {
            [2. / 3., 0., -1. / 3.]
        } else {
            [1., 0., -0.5]
        };
        for (i, r) in result.iter().enumerate() {
            assert!((r.value - expected[i]).abs() < 1e-13);
            assert_eq!(r.pairs, 3 - i);
        }
    }
    let mut opt = TimeSeriesOptions::new();
    opt = opt.with_center(scioxide_stats::TimeSeriesCentering::KnownZero);
    let skip = StatisticsOptions::new().with_missing_policy(MissingPolicy::Skip);
    let x = [1., f64::NAN, 3., 4.];
    let r = time_series_summary_with_options(
        &x,
        Some(&[true, false, true, true]),
        TimeSeriesStatistic::Covariance,
        2,
        opt,
        skip,
    )
    .unwrap();
    assert_eq!(r[0].pairs, 3);
    assert_eq!(r[1].pairs, 1);
    assert_eq!(r[2].pairs, 1);
    assert!((r[1].value - 4.).abs() < 1e-14);
    opt = opt.with_normalization(LagNormalization::PairCount);
    let r = time_series_summary_with_options(
        &x,
        Some(&[true, false, true, true]),
        TimeSeriesStatistic::Covariance,
        2,
        opt,
        skip,
    )
    .unwrap();
    assert_eq!(r[1].value, 12.);
    let options = StatisticsOptions::new();
    let req = TimeSeriesWorkspace::scalar_requirements(4, 2, options).unwrap();
    let mut w = TimeSeriesWorkspace::workspace(&req).unwrap();
    let mut out = vec![
        TimeSeriesLag {
            value: 9.,
            pairs: 99
        };
        3
    ];
    assert!(
        time_series_summary_into(
            &[1., 1., 1., 1.],
            None,
            TimeSeriesStatistic::Acf,
            2,
            TimeSeriesOptions::new(),
            options,
            &mut out,
            &mut w,
            ExecutionContext::sequential()
        )
        .is_err()
    );
    assert_eq!(out[0].value, 9.);
    assert_eq!(out[2].pairs, 99);
    for (samples, lag, bytes) in [(0, 0, usize::MAX), (4, 4, usize::MAX), (4, 2, 0)] {
        assert!(
            TimeSeriesWorkspace::<f64>::scalar_requirements(
                samples,
                lag,
                options.with_max_bytes(bytes)
            )
            .is_err()
        );
    }
}
fn complex_summaries<S: RealScalar>()
where
    Complex<S>: NumericScalar<Real = S>,
{
    let z = S::ZERO;
    let o = S::ONE;
    let x = [
        Complex::new(o, z),
        Complex::new(z, o),
        Complex::new(-o, z),
        Complex::new(z, -o),
    ];
    let acf = time_series_summary_with_options(
        &x,
        None,
        TimeSeriesStatistic::Acf,
        2,
        TimeSeriesOptions::new(),
        StatisticsOptions::new(),
    )
    .unwrap();
    let pacf = time_series_summary_with_options(
        &x,
        None,
        TimeSeriesStatistic::Pacf,
        2,
        TimeSeriesOptions::new(),
        StatisticsOptions::new(),
    )
    .unwrap();
    assert!((acf[1].value.im.into_f64() - 0.75).abs() < 2e-6);
    assert!(acf[1].value.re.abs() < S::EPSILON);
    assert!((pacf[2].value.re.into_f64() - 1. / 7.).abs() < 2e-6);
}
#[test]
fn complex_orientation_and_hermitian_pacf_both_widths() {
    complex_summaries::<f32>();
    complex_summaries::<f64>();
}
#[test]
fn extended_range_correlations_do_not_require_representable_covariance() {
    let x = [-1e300, 0., 1e300];
    let r = time_series_summary_with_options(
        &x,
        None,
        TimeSeriesStatistic::Acf,
        2,
        TimeSeriesOptions::new(),
        StatisticsOptions::new(),
    )
    .unwrap();
    assert!((r[2].value + 0.5).abs() < 1e-14);
    assert!(
        time_series_summary_with_options(
            &x,
            None,
            TimeSeriesStatistic::Covariance,
            2,
            TimeSeriesOptions::new(),
            StatisticsOptions::new()
        )
        .is_err()
    );
}
#[test]
fn signal_periodogram_parseval_and_zero_power() {
    use scioxide_signal::{SpectralEstimateOptions, SpectralEstimatePlan, StftPacking};
    let options = SpectralEstimateOptions::new()
        .with_fft_len(Some(8))
        .with_sample_rate(8.)
        .with_packing(StftPacking::OneSided)
        .with_detrend(None);
    let plan =
        SpectralEstimatePlan::<f64>::build(&[1.; 8], options, ExecutionContext::sequential())
            .unwrap();
    let mut work = PeriodogramWorkspace::new(&plan, StatisticsOptions::new()).unwrap();
    let mut out = vec![PeriodogramBin::default(); plan.bins()];
    let report = periodogram_diagnostics_into(
        &plan,
        &[1., 0., -1., 0., 1., 0., -1., 0.],
        &mut out,
        &mut work,
        ExecutionContext::sequential(),
    )
    .unwrap();
    assert!((report.integrated_power - 0.5).abs() < 1e-12);
    assert_eq!(report.peak_frequency, Some(2.));
    assert_eq!(out.last().unwrap().cumulative_power, Some(1.));
    let report = periodogram_diagnostics_into(
        &plan,
        &[0.; 8],
        &mut out,
        &mut work,
        ExecutionContext::sequential(),
    )
    .unwrap();
    assert_eq!(report.peak_frequency, None);
    assert!(out.iter().all(|v| v.cumulative_power.is_none()));
}

use scioxide_stats::{FitObjective, MissingPolicy};
fn close<S: RealScalar>(value: S, expected: f64) {
    let tol = if S::EPSILON.into_f64() > 1e-10 {
        2e-4
    } else {
        2e-11
    };
    assert!(
        (value.into_f64() - expected).abs() <= tol * (1. + expected.abs()),
        "{} != {} (tol {})",
        value.into_f64(),
        expected,
        tol
    );
}
fn rational(s: &str) -> f64 {
    let mut v = s.split('/');
    let a: f64 = v.next().unwrap().parse().unwrap();
    a / v.next().map_or(1., |b| b.parse::<f64>().unwrap())
}
fn oracle<S: RealScalar>()
where
    Complex<S>: NumericScalar<Real = S>,
{
    let mut lines = include_str!("data/time_series.tsv")
        .lines()
        .filter(|l| !l.starts_with('#'));
    while let Some(case) = lines.next() {
        let parts: Vec<_> = case.split('\t').collect();
        assert_eq!(parts[0], "case");
        let order = ArimaOrder {
            p: parts[2].parse().unwrap(),
            d: parts[3].parse().unwrap(),
            q: parts[4].parse().unwrap(),
        };
        let parameters: Vec<_> = parts[5]
            .split(',')
            .map(|v| S::try_from_f64_lossy(rational(v)).unwrap())
            .collect();
        let values: Vec<_> = parts[6]
            .split(',')
            .map(|v| S::try_from_f64_lossy(rational(v)).unwrap())
            .collect();
        let valid: Vec<_> = parts[7].bytes().map(|b| b == b'1').collect();
        let problem = PreparedArimaFit::new(
            &values,
            Some(&valid),
            order,
            ArimaTrend::None,
            StatisticsOptions::new().with_missing_policy(MissingPolicy::Skip),
        )
        .unwrap();
        let mut work = problem.workspace().unwrap();
        let mut innovations = vec![ArimaInnovation::default(); values.len()];
        let report = problem
            .evaluate_into(
                &parameters,
                &mut innovations,
                &mut work,
                ExecutionContext::sequential(),
            )
            .unwrap();
        let model = problem
            .model(&parameters, &mut work, ExecutionContext::sequential())
            .unwrap();
        let mut forecasts = vec![
            ArimaForecast {
                mean: S::ZERO,
                variance: S::ZERO
            };
            4
        ];
        model
            .forecast_into(&mut forecasts, &mut work, ExecutionContext::sequential())
            .unwrap();
        for line in lines.by_ref() {
            let fields: Vec<_> = line.split('\t').collect();
            match fields[0] {
                "innovation" => {
                    let t: usize = fields[1].parse().unwrap();
                    let innovation = innovations[t];
                    close(innovation.prediction.unwrap(), rational(fields[2]));
                    close(innovation.variance.unwrap(), rational(fields[3]));
                    if fields[4] == "-" {
                        assert!(innovation.residual.is_none());
                    } else {
                        close(innovation.residual.unwrap(), rational(fields[4]));
                    }
                }
                "forecast" => {
                    let t: usize = fields[1].parse().unwrap();
                    close(forecasts[t].mean, rational(fields[2]));
                    close(forecasts[t].variance, rational(fields[3]));
                }
                "nll" => close(report.negative_log_likelihood, fields[1].parse().unwrap()),
                "end" => break,
                _ => unreachable!("unknown fixture row"),
            }
        }
        assert_eq!(
            report.observations,
            valid[order.d..].iter().filter(|&&b| b).count()
        );
        // Reset after forecast state changes: the objective is history-independent.
        close(
            problem.value(&parameters, &mut work).unwrap(),
            report.negative_log_likelihood.into_f64(),
        );
    }
}
#[test]
fn independent_exact_gaussian_fixtures() {
    oracle::<f32>();
    oracle::<f64>();
}
#[test]
fn trends_stationarity_and_original_unit_forecasts() {
    let data = [10., 12., 14., 16., 18.];
    let order = ArimaOrder { p: 0, d: 1, q: 0 };
    let problem = PreparedArimaFit::new(
        &data,
        None,
        order,
        ArimaTrend::Linear,
        StatisticsOptions::new(),
    )
    .unwrap();
    let mut work = problem.workspace().unwrap();
    let model = problem
        .model(&[2., 0.25], &mut work, ExecutionContext::sequential())
        .unwrap();
    let mut forecast = [ArimaForecast {
        mean: 0.,
        variance: 0.,
    }; 3];
    model
        .forecast_into(&mut forecast, &mut work, ExecutionContext::sequential())
        .unwrap();
    for (i, x) in forecast.iter().enumerate() {
        close(x.mean, 20. + 2. * i as f64);
        close(x.variance, 0.25 * (i + 1) as f64);
    }
    assert!(
        PreparedArimaFit::new(
            &data,
            None,
            order,
            ArimaTrend::Constant,
            StatisticsOptions::new()
        )
        .is_err()
    );
    assert!(
        PreparedArimaFit::new(
            &data,
            Some(&[false, true, true, true, true]),
            order,
            ArimaTrend::None,
            StatisticsOptions::new().with_missing_policy(MissingPolicy::Skip)
        )
        .is_err()
    );
    for (p, q) in [(1, 0), (0, 1)] {
        let problem = PreparedArimaFit::new(
            &data,
            None,
            ArimaOrder { p, d: 0, q },
            ArimaTrend::None,
            StatisticsOptions::new(),
        )
        .unwrap();
        let mut w = problem.workspace().unwrap();
        for coefficient in [-1.01, -1., 1., 1.01] {
            assert!(problem.value(&[coefficient, 1.], &mut w).is_err());
        }
        for coefficient in [-0.999, 0.999] {
            assert!(problem.value(&[coefficient, 1.], &mut w).is_ok());
        }
    }
    let problem = PreparedArimaFit::new(
        &data,
        None,
        ArimaOrder { p: 1, d: 0, q: 0 },
        ArimaTrend::None,
        StatisticsOptions::new(),
    )
    .unwrap();
    let mut work = problem.workspace().unwrap();
    let model = problem
        .model(&[0.5, 1.], &mut work, ExecutionContext::sequential())
        .unwrap();
    close(model.spectral_density(0., 1.).unwrap(), 4.);
    close(model.spectral_density(0.5, 1.).unwrap(), 4. / 9.);
}

mod solver {
    use crate as scioxide_stats;
    include!("support/fit_solver.rs");
}
fn identity() -> scioxide_core::SolveIdentity {
    use scioxide_core::*;
    SolveIdentity::new(
        ProblemToken::new(556),
        scioxide_core::algorithm_id!("stats", "arima", 1),
        0,
        0,
    )
}
#[test]
fn injected_fitting_verification_and_budgets() {
    use scioxide_core::*;
    let data = [1., 3., 1., 3.];
    let problem = PreparedArimaFit::new(
        &data,
        None,
        ArimaOrder::default(),
        ArimaTrend::Constant,
        StatisticsOptions::new(),
    )
    .unwrap();
    let bounds = Bounds::new(vec![(Some(1.), Some(3.)), (Some(0.5), Some(1.5))]).unwrap();
    let mut solver = solver::GridSolver { intervals: 4 };
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut ctx = state
        .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
        .unwrap();
    let fit = problem
        .fit_scoped(&[1., 0.5], &bounds, &mut solver, &mut ctx)
        .unwrap();
    assert_eq!(
        fit.report().termination(),
        scioxide_stats::StatsTermination::Converged
    );
    let model = fit.model().unwrap();
    close(model.parameters()[0], 2.);
    close(model.parameters()[1], 1.);
    close(
        fit.report().objective().unwrap(),
        2. * (1. + (2. * core::f64::consts::PI).ln()),
    );
    assert_eq!(
        fit.report()
            .counts()
            .total()
            .unwrap()
            .provider(EvaluationRole::Objective),
        26
    );
    let mut w = scioxide_stats::FitWorkspace::for_objective(
        &problem,
        scioxide_stats::StatisticsOptions::new(),
    )
    .unwrap();
    let mut output = [123.; 2];
    let mut state = SolveState::new();
    let mut budgets =
        SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, Budget::try_from(1).unwrap());
    let mut ctx = state
        .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
        .unwrap();
    let report = scioxide_stats::fit_into(
        scioxide_stats::FitRequest {
            objective: &problem,
            derivatives: None,
            bounds: &bounds,
            integrality: &[false; 2],
            initial: &[2., 1.],
        },
        &mut solver,
        &mut output,
        &mut w,
        &mut ctx,
    )
    .unwrap();
    assert_eq!(
        report.termination(),
        scioxide_stats::StatsTermination::BudgetExhausted
    );
    assert!(report.objective().is_none());
    assert_eq!(output, [123.; 2]);
    let token = CancellationToken::new();
    token.cancel();
    let mut iw = problem.workspace().unwrap();
    let mut out = vec![
        ArimaInnovation {
            prediction: Some(99.),
            variance: None,
            residual: None
        };
        4
    ];
    assert!(
        problem
            .evaluate_into(
                &[2., 1.],
                &mut out,
                &mut iw,
                ExecutionContext::sequential().with_cancellation(&token)
            )
            .is_err()
    );
    assert_eq!(out[0].prediction, Some(99.));
    assert!(problem.workspace().is_ok());
    assert!(
        scioxide_stats::FitWorkspace::for_objective(
            &problem,
            scioxide_stats::StatisticsOptions::new()
                .with_max_bytes(problem.workspace_bytes().unwrap())
        )
        .is_err()
    );
}

#[cfg(feature = "std")]
#[test]
fn strided_summary_and_typed_objective_executors() {
    use core::num::NonZeroUsize;
    use scioxide_core::*;
    let shape = [6, 2];
    let strides = [1, 6];
    let layout = StridedLayout::new(&shape, &strides).unwrap();
    let data = [1., 2., 1., -1., 0., 1., 2., 4., 2., -2., 0., 2.];
    let pool = ThreadExecutor::new(NonZeroUsize::new(2).unwrap());
    let execution = scioxide_stats::StatisticsOptions::new()
        .with_execution_policy(ExecutionPolicy::Parallel(NonZeroUsize::new(2).unwrap()));
    let context = ExecutionContext::new(&pool);
    let view = BatchView::new(&data, layout, Axis::Index(0)).unwrap();
    let req = TimeSeriesWorkspace::workspace_requirements(view, 3, execution, context).unwrap();
    let mut w = TimeSeriesWorkspace::workspace(&req).unwrap();
    let mut parallel = [TimeSeriesLag::default(); 8];
    time_series_summaries_into(
        view,
        TimeSeriesStatistic::Acf,
        3,
        TimeSeriesOptions::new(),
        execution,
        &mut parallel,
        &mut w,
        context,
    )
    .unwrap();
    for lane in 0..2 {
        let expected = time_series_summary_with_options(
            &data[lane * 6..lane * 6 + 6],
            None,
            TimeSeriesStatistic::Acf,
            3,
            TimeSeriesOptions::new(),
            StatisticsOptions::new(),
        )
        .unwrap();
        for j in 0..4 {
            assert_eq!(parallel[lane * 4 + j].value, expected[j].value);
        }
    }
    let problem = PreparedArimaFit::new(
        &data,
        None,
        ArimaOrder { p: 1, d: 0, q: 1 },
        ArimaTrend::None,
        StatisticsOptions::new(),
    )
    .unwrap();
    let parameters = [0.5, 0.25, 1., 0.25, 0.5, 1., -0.5, 0.25, 1.];
    let mut work = [problem.workspace().unwrap(), problem.workspace().unwrap()];
    let mut output = [0.; 3];
    let mut stage = [0.; 3];
    scioxide_stats::fit_objectives_into(
        &problem,
        &parameters,
        execution,
        &mut output,
        &mut stage,
        &mut work,
        context,
    )
    .unwrap();
    for (i, p) in parameters.chunks_exact(3).enumerate() {
        assert_eq!(output[i], problem.value(p, &mut work[0]).unwrap());
    }
}
#[test]
fn borrowed_arima_views_and_invalid_inputs() {
    use scioxide_core::StridedLayout;
    let data = [99., 1., 99., 2., 99., -1., 99., 0.5];
    let shape = [4, 2];
    let strides = [2, 1];
    let layout = StridedLayout::new(&shape, &strides).unwrap();
    let order = ArimaOrder { p: 1, d: 0, q: 1 };
    let series = BatchView::new(&data, layout, Axis::Index(0)).unwrap();
    let view =
        PreparedArimaFit::from_view(series, 1, order, ArimaTrend::None, StatisticsOptions::new())
            .unwrap();
    for lane in [2, usize::MAX] {
        assert!(
            PreparedArimaFit::from_view(
                series,
                lane,
                order,
                ArimaTrend::None,
                StatisticsOptions::new()
            )
            .is_err()
        );
    }
    let contiguous = PreparedArimaFit::new(
        &[1., 2., -1., 0.5],
        None,
        order,
        ArimaTrend::None,
        StatisticsOptions::new(),
    )
    .unwrap();
    let mut w = view.workspace().unwrap();
    assert_eq!(
        view.value(&[0.5, 0.25, 1.], &mut w).unwrap(),
        contiguous.value(&[0.5, 0.25, 1.], &mut w).unwrap()
    );
    for params in [
        &[0.5, 0.25, 0.][..],
        &[0.5, f64::NAN, 1.],
        &[1.5, 0.25, 1.],
        &[0.5, 0.25][..],
    ] {
        assert!(view.value(params, &mut w).is_err());
    }
    let wrong = ArimaOrder { p: 2, d: 0, q: 0 };
    let mut wrong = PreparedArimaFit::new(
        &[1., 2., -1., 0.5],
        None,
        wrong,
        ArimaTrend::None,
        StatisticsOptions::new(),
    )
    .unwrap()
    .workspace()
    .unwrap();
    assert!(view.value(&[0.5, 0.25, 1.], &mut wrong).is_err());
    assert!(
        PreparedArimaFit::new(
            &[f64::NAN],
            None,
            ArimaOrder::default(),
            ArimaTrend::None,
            StatisticsOptions::new()
        )
        .is_err()
    );
    assert!(
        PreparedArimaFit::new(
            &[0.],
            Some(&[false]),
            ArimaOrder::default(),
            ArimaTrend::None,
            StatisticsOptions::new().with_missing_policy(MissingPolicy::Skip)
        )
        .is_err()
    );
    let p = PreparedArimaFit::new(
        &[1_f32, 2., 3.],
        None,
        ArimaOrder { p: 1, d: 0, q: 0 },
        ArimaTrend::None,
        StatisticsOptions::new(),
    )
    .unwrap();
    let mut w = p.workspace().unwrap();
    assert!(p.value(&[1. - f32::EPSILON, 1.], &mut w).is_err());
    assert!(p.value(&[0.99, 1.], &mut w).is_ok());
}
fn injected_arma<S: RealScalar>()
where
    Complex<S>: NumericScalar<Real = S>,
{
    use scioxide_core::*;
    let s = |v| S::try_from_f64_lossy(v).unwrap();
    let order = ArimaOrder { p: 1, d: 1, q: 1 };
    let data = [0., 1., 2., 1., 3., 2., 1., 2.].map(s);
    let problem = PreparedArimaFit::new(
        &data,
        None,
        order,
        ArimaTrend::None,
        StatisticsOptions::new(),
    )
    .unwrap();
    let bounds = Bounds::new(vec![
        (Some(s(-0.5)), Some(s(0.5))),
        (Some(s(-0.5)), Some(s(0.5))),
        (Some(s(0.5)), Some(s(2.5))),
    ])
    .unwrap();
    let start = [s(0.5), s(0.5), s(0.5)];
    let mut work = problem.workspace().unwrap();
    let initial = problem.value(&start, &mut work).unwrap();
    let mut solver = solver::GridSolver { intervals: 4 };
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
        .unwrap();
    let fit = problem
        .fit_scoped(&start, &bounds, &mut solver, &mut context)
        .unwrap();
    assert!(fit.report().objective().unwrap() < initial);
    let model = fit.model().unwrap();
    close(
        problem.value(model.parameters(), &mut work).unwrap(),
        fit.report().objective().unwrap().into_f64(),
    );
    let mut forecast = [ArimaForecast {
        mean: S::ZERO,
        variance: S::ZERO,
    }; 3];
    model
        .forecast_into(&mut forecast, &mut work, ExecutionContext::sequential())
        .unwrap();
    assert!(
        forecast
            .iter()
            .all(|x| x.mean.is_finite() && x.variance > S::ZERO)
    );
}
#[test]
fn injected_arma_arima_fitting_both_widths() {
    injected_arma::<f32>();
    injected_arma::<f64>();
}
#[test]
fn higher_order_complex_pacf_matches_independent_dense_yule_walker() {
    use scioxide_linalg::{LuDecomposition, Matrix, Vector};
    let data: Vec<_> = (0..16)
        .map(|i| Complex::new(((i * 7) % 11) as f64 - 5., ((i * 3) % 7) as f64 - 3.))
        .collect();
    let acf = time_series_summary_with_options(
        &data,
        None,
        TimeSeriesStatistic::Acf,
        6,
        TimeSeriesOptions::new(),
        StatisticsOptions::new(),
    )
    .unwrap();
    let pacf = time_series_summary_with_options(
        &data,
        None,
        TimeSeriesStatistic::Pacf,
        6,
        TimeSeriesOptions::new(),
        StatisticsOptions::new(),
    )
    .unwrap();
    for k in 1..=6 {
        let mut a = Matrix::zeros(k, k).unwrap();
        for i in 0..k {
            for j in 0..k {
                a.view_mut()
                    .set(
                        i,
                        j,
                        if i >= j {
                            acf[i - j].value
                        } else {
                            acf[j - i].value.conj()
                        },
                    )
                    .unwrap();
            }
        }
        let b = Vector::from_vec(acf[1..=k].iter().map(|x| x.value).collect());
        let solution = LuDecomposition::factor(&a).unwrap().solve(&b).unwrap();
        let expected = solution.view().get(k - 1).unwrap();
        assert!((expected - pacf[k].value).abs() < 1e-12);
    }
}
#[test]
fn supplied_innovations_simulation_reuses_signal_filter() {
    use scioxide_signal::{
        IirDomain, IirFilter, IirRequest, IirState, SignalExecutionOptions, SignalLayout,
        TransferFunction,
    };
    let transfer = TransferFunction::new(IirDomain::Digital, &[1.], &[1., -0.5]).unwrap();
    let filter = IirFilter::from_transfer(&transfer, SignalExecutionOptions::new()).unwrap();
    let noise: Vec<f64> = (0..32).map(|i| ((i * 7) % 11) as f64 / 4. - 1.25).collect();
    let mut samples = vec![0.; noise.len()];
    let mut state = IirState::new(&filter, 1).unwrap();
    let mut sw = filter
        .workspace(IirRequest::Filter(SignalLayout::Contiguous(noise.len())))
        .unwrap();
    filter
        .filter_into(
            &noise,
            &mut samples,
            &mut state,
            &mut sw,
            ExecutionContext::sequential(),
        )
        .unwrap();
    let problem = PreparedArimaFit::new(
        &samples,
        None,
        ArimaOrder { p: 1, d: 0, q: 0 },
        ArimaTrend::None,
        StatisticsOptions::new(),
    )
    .unwrap();
    let mut work = problem.workspace().unwrap();
    let mut output = vec![ArimaInnovation::default(); samples.len()];
    problem
        .evaluate_into(
            &[0.5, 1.],
            &mut output,
            &mut work,
            ExecutionContext::sequential(),
        )
        .unwrap();
    close(output[0].variance.unwrap(), 4. / 3.);
    for (actual, &expected) in output[1..].iter().zip(&noise[1..]) {
        close(actual.residual.unwrap(), expected);
        close(actual.variance.unwrap(), 1.);
    }
}
#[test]
fn centering_preserves_small_differences_and_exact_constants() {
    let x = [1e16, 1e16 + 2., 1e16 + 4., 1e16 + 6.];
    let result = time_series_summary_with_options(
        &x,
        None,
        TimeSeriesStatistic::Covariance,
        3,
        TimeSeriesOptions::new(),
        StatisticsOptions::new(),
    )
    .unwrap();
    for (actual, expected) in result.iter().zip([5., 1.25, -1.5, -2.25]) {
        close(actual.value, expected);
    }
    assert!(
        time_series_summary_with_options(
            &[0.1; 6],
            None,
            TimeSeriesStatistic::Acf,
            2,
            TimeSeriesOptions::new(),
            StatisticsOptions::new()
        )
        .is_err()
    );
}
#[test]
fn f32_centering_preserves_sub_ulp_mean_offsets() {
    let values = [10_000_000_f32, 10_000_001., 10_000_002., 10_000_003.];
    let result = time_series_summary_with_options(
        &values,
        None,
        TimeSeriesStatistic::Covariance,
        3,
        TimeSeriesOptions::new(),
        StatisticsOptions::new(),
    )
    .unwrap();
    for (actual, expected) in result.iter().zip([1.25, 0.3125, -0.375, -0.5625]) {
        close(actual.value, expected);
    }
    assert!(
        time_series_summary_with_options(
            &[0.1_f32; 6],
            None,
            TimeSeriesStatistic::Pacf,
            2,
            TimeSeriesOptions::new(),
            StatisticsOptions::new()
        )
        .is_err()
    );
}

fn skip() -> StatisticsOptions {
    StatisticsOptions::new().with_missing_policy(MissingPolicy::Skip)
}

#[test]
fn real_summary_centering_retains_cancellation_and_tiny_covariances() {
    // The exact mean is 1/3, despite the cancellation in the shifted first pass.
    let result = time_series_summary_with_options(
        &[1e16, 1., -1e16],
        None,
        TimeSeriesStatistic::Covariance,
        1,
        TimeSeriesOptions::new(),
        StatisticsOptions::new(),
    )
    .unwrap();
    close(result[1].value, -4. / 27.);
    // Native binary64 products must not be projected to f32 before division.
    let result = time_series_summary_with_options(
        &[1e19_f32, -1e19, 1e19, -1e19],
        None,
        TimeSeriesStatistic::Covariance,
        1,
        TimeSeriesOptions::new(),
        StatisticsOptions::new(),
    )
    .unwrap();
    let square = f64::from(1e19_f32).powi(2);
    close(result[0].value, square);
    close(result[1].value, -0.75 * square);
    // Exponent 64 is admitted natively, but its variance exceeds f32::MAX.
    let magnitude = 2.0_f32.powi(64);
    let values = [magnitude, -magnitude];
    let result = time_series_summary_with_options(
        &values,
        None,
        TimeSeriesStatistic::Acf,
        1,
        TimeSeriesOptions::new(),
        StatisticsOptions::new(),
    )
    .unwrap();
    close(result[1].value, -0.5);
    assert!(
        time_series_summary_with_options(
            &values,
            None,
            TimeSeriesStatistic::Covariance,
            1,
            TimeSeriesOptions::new(),
            StatisticsOptions::new()
        )
        .is_err()
    );
    let result = time_series_summary_with_options(
        &[0., f64::from_bits(2)],
        None,
        TimeSeriesStatistic::Covariance,
        1,
        TimeSeriesOptions::new(),
        StatisticsOptions::new(),
    )
    .unwrap();
    assert_eq!(result[0].value, 0.);
    assert!(!result[0].value.is_sign_negative());
    assert_eq!(result[1].value, 0.);
    assert!(result[1].value.is_sign_negative());
}

/// Returns the shortfall of a `ResourceLimit` error after checking `required > limit`.
fn resource<T>(result: Result<T, scioxide_core::SciOxideError>) -> (&'static str, usize, usize) {
    let Some(error) = result.err() else {
        panic!("expected a resource limit");
    };
    let scioxide_core::SciOxideErrorView::ResourceLimit {
        resource,
        required,
        limit,
    } = error.view()
    else {
        panic!("expected a resource limit, got {error}");
    };
    assert!(required > limit, "{error}");
    (resource.as_str(), required, limit)
}

fn same_lags(actual: &[TimeSeriesLag<f64>], expected: &[TimeSeriesLag<f64>]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual.value.to_bits(), expected.value.to_bits());
        assert_eq!(actual.pairs, expected.pairs);
    }
}

#[test]
fn batch_summaries_honor_every_statistics_option() {
    use scioxide_core::{
        Capacity, NonFiniteOrigin, SciOxideErrorKind, SciOxideErrorView, StridedLayout,
    };
    // Two records along axis 0: [1, NaN, 3, 5, 2] and [2, 4, 1, inf, 3].
    let shape = [5, 2];
    let strides = [2, 1];
    let layout = StridedLayout::new(&shape, &strides).unwrap();
    let data = [1., 2., f64::NAN, 4., 3., 1., 5., f64::INFINITY, 2., 3.];
    let mask = [true, false, true, true, true, true, true, true, true, true];
    let view = BatchView::new(&data, layout, Axis::Index(0))
        .unwrap()
        .with_validity(&mask)
        .unwrap();
    let summary = TimeSeriesOptions::new();
    let context = ExecutionContext::sequential();
    let run = |options: StatisticsOptions, workspace: &mut TimeSeriesWorkspace<f64>| {
        let mut output = [TimeSeriesLag::default(); 4];
        time_series_summaries_into(
            view,
            TimeSeriesStatistic::Covariance,
            1,
            summary,
            options,
            &mut output,
            workspace,
            context,
        )
        .map(|()| output)
    };
    let both = skip().with_non_finite_policy(NonFinitePolicy::Skip);
    let req = TimeSeriesWorkspace::workspace_requirements(view, 1, both, context).unwrap();
    assert_eq!((req.samples(), req.lags(), req.lanes()), (5, 2, 2));
    let mut workspace = TimeSeriesWorkspace::workspace(&req).unwrap();

    // Missing observations are rejected unless skipped.
    let missing = run(StatisticsOptions::new(), &mut workspace).unwrap_err();
    assert_eq!(missing.kind(), SciOxideErrorKind::InvalidInput);
    // Skipping missing values still rejects a present infinity as an input.
    let non_finite = run(skip(), &mut workspace).unwrap_err();
    assert!(matches!(
        non_finite.view(),
        SciOxideErrorView::NonFinite {
            origin: NonFiniteOrigin::Input,
            ..
        }
    ));
    // Skipping both leaves gaps on the grid, exactly as masking the infinity.
    let gaps = run(both, &mut workspace).unwrap();
    let records = [
        ([1., f64::NAN, 3., 5., 2.], [true, false, true, true, true]),
        ([2., 4., 1., 0., 3.], [true, true, true, false, true]),
    ];
    for (lane, (values, mask)) in records.iter().enumerate() {
        let expected = time_series_summary_with_options(
            values,
            Some(mask),
            TimeSeriesStatistic::Covariance,
            1,
            summary,
            skip(),
        )
        .unwrap();
        same_lags(&gaps[lane * 2..lane * 2 + 2], &expected);
    }
    assert_eq!((gaps[0].pairs, gaps[1].pairs), (4, 2));

    // Byte limits apply to requirements and to retained workspace capacity.
    let bytes = req.required_bytes();
    assert_eq!(
        resource(TimeSeriesWorkspace::<f64>::workspace_requirements(
            view,
            1,
            both.with_max_bytes(bytes - 1),
            context
        )),
        ("statistics_workspace_bytes", bytes, bytes - 1)
    );
    let scalar = TimeSeriesWorkspace::scalar_requirements(5, 1, both).unwrap();
    let mut grown = TimeSeriesWorkspace::workspace(&req).unwrap();
    assert!(grown.check(&scalar).unwrap() == Capacity::Sufficient);
    assert_eq!(
        resource(
            grown.check(
                &TimeSeriesWorkspace::scalar_requirements(
                    5,
                    1,
                    both.with_max_bytes(scalar.required_bytes())
                )
                .unwrap()
            )
        )
        .0,
        "statistics_workspace_bytes"
    );
    let sentinel = [TimeSeriesLag {
        value: 7.,
        pairs: 7,
    }; 2];
    let mut output = sentinel;
    let values = [1., 2., 3., 4., 5.];
    let limited = both.with_max_bytes(scalar.required_bytes());
    let (name, _, limit) = resource(time_series_summary_into(
        &values,
        None,
        TimeSeriesStatistic::Acf,
        1,
        summary,
        limited,
        &mut output,
        &mut grown,
        context,
    ));
    assert_eq!(
        (name, limit),
        ("statistics_workspace_bytes", scalar.required_bytes())
    );
    same_lags(&output, &sentinel);

    // An undersized workspace names its first short capacity; reserving fixes it.
    let short = TimeSeriesWorkspace::scalar_requirements(5, 0, both).unwrap();
    let mut small = TimeSeriesWorkspace::workspace(&short).unwrap();
    assert_eq!(small.check(&req).unwrap(), Capacity::NeedsGrowth);
    assert_eq!(resource(run(both, &mut small)), ("time_series_lags", 2, 1));
    small.try_reserve(&req).unwrap();
    assert_eq!(small.check(&req).unwrap(), Capacity::Sufficient);
    same_lags(&run(both, &mut small).unwrap(), &gaps);
}

#[test]
fn scalar_summaries_skip_non_finite_values_as_gaps_and_limit_owned_bytes() {
    let values = [1., f64::NAN, -2., f64::INFINITY, 4., 3.];
    let mask = [true, false, true, false, true, true];
    let summary = TimeSeriesOptions::new();
    for statistic in [
        TimeSeriesStatistic::Covariance,
        TimeSeriesStatistic::Acf,
        TimeSeriesStatistic::Pacf,
    ] {
        let skipped = time_series_summary_with_options(
            &values,
            None,
            statistic,
            2,
            summary,
            StatisticsOptions::new().with_non_finite_policy(NonFinitePolicy::Skip),
        )
        .unwrap();
        let masked =
            time_series_summary_with_options(&values, Some(&mask), statistic, 2, summary, skip())
                .unwrap();
        same_lags(&skipped, &masked);
        assert!(
            time_series_summary_with_options(
                &values,
                None,
                statistic,
                2,
                summary,
                StatisticsOptions::new()
            )
            .is_err()
        );
    }
    // The owned form charges its workspace and the returned lags together.
    let workspace =
        TimeSeriesWorkspace::<f64>::scalar_requirements(values.len(), 2, StatisticsOptions::new())
            .unwrap()
            .required_bytes();
    let total = workspace + 3 * core::mem::size_of::<TimeSeriesLag<f64>>();
    let finite = [1., 2., -1., 4., 3., 0.];
    let options = StatisticsOptions::new();
    assert_eq!(
        resource(time_series_summary_with_options(
            &finite,
            None,
            TimeSeriesStatistic::Acf,
            2,
            summary,
            options.with_max_bytes(total - 1)
        )),
        ("statistics_workspace_bytes", total, total - 1)
    );
    assert!(
        time_series_summary_with_options(
            &finite,
            None,
            TimeSeriesStatistic::Acf,
            2,
            summary,
            options.with_max_bytes(total)
        )
        .is_ok()
    );
}

#[test]
fn periodogram_workspace_rejects_skips_and_honors_byte_limits() {
    use scioxide_core::SciOxideErrorKind;
    use scioxide_signal::{SpectralEstimateOptions, SpectralEstimatePlan};
    let plan = SpectralEstimatePlan::<f64>::build(
        &[1.; 8],
        SpectralEstimateOptions::new()
            .with_fft_len(Some(8))
            .with_sample_rate(8.),
        ExecutionContext::sequential(),
    )
    .unwrap();
    for options in [
        skip(),
        StatisticsOptions::new().with_non_finite_policy(NonFinitePolicy::Skip),
    ] {
        assert_eq!(
            PeriodogramWorkspace::new(&plan, options)
                .err()
                .unwrap()
                .kind(),
            SciOxideErrorKind::InvalidInput
        );
    }
    let bytes = PeriodogramWorkspace::required_bytes(&plan).unwrap();
    assert_eq!(
        resource(PeriodogramWorkspace::new(
            &plan,
            StatisticsOptions::new().with_max_bytes(bytes - 1)
        )),
        ("statistics_workspace_bytes", bytes, bytes - 1)
    );
    assert!(
        PeriodogramWorkspace::new(&plan, StatisticsOptions::new().with_max_bytes(bytes)).is_ok()
    );
}

#[test]
fn arima_preparation_honors_statistics_options_and_retained_limits() {
    use scioxide_core::{NonFiniteOrigin, SciOxideErrorView, StridedLayout};
    let order = ArimaOrder { p: 1, d: 0, q: 0 };
    let values = [0.5, f64::NAN, 0.75, 0.0, -0.5];
    let mask = [true, false, true, true, true];
    let parameters = [0.5, 1.0];
    let context = ExecutionContext::sequential();
    let rejected = PreparedArimaFit::new(
        &values,
        None,
        order,
        ArimaTrend::None,
        StatisticsOptions::new(),
    );
    assert!(matches!(
        rejected.err().unwrap().view(),
        SciOxideErrorView::NonFinite {
            origin: NonFiniteOrigin::Input,
            ..
        }
    ));
    assert!(
        PreparedArimaFit::new(
            &values,
            Some(&mask),
            order,
            ArimaTrend::None,
            StatisticsOptions::new()
        )
        .is_err()
    );
    let skipped = PreparedArimaFit::new(
        &values,
        None,
        order,
        ArimaTrend::None,
        StatisticsOptions::new().with_non_finite_policy(NonFinitePolicy::Skip),
    )
    .unwrap();
    let masked =
        PreparedArimaFit::new(&values, Some(&mask), order, ArimaTrend::None, skip()).unwrap();
    let evaluate = |problem: &PreparedArimaFit<'_, f64>| {
        let mut workspace = problem.workspace().unwrap();
        let mut innovations = [ArimaInnovation::default(); 5];
        let likelihood = problem
            .evaluate_into(&parameters, &mut innovations, &mut workspace, context)
            .unwrap();
        (likelihood, innovations)
    };
    let (skip_likelihood, skip_innovations) = evaluate(&skipped);
    let (mask_likelihood, mask_innovations) = evaluate(&masked);
    assert_eq!(skip_likelihood.observations, 4);
    assert_eq!(
        skip_likelihood.negative_log_likelihood,
        mask_likelihood.negative_log_likelihood
    );
    assert!(skip_innovations[1].residual.is_none() && mask_innovations[1].residual.is_none());

    // A lane of a masked view reads its own mask in sample order.
    let shape = [5, 2];
    let strides = [1, 5];
    let layout = StridedLayout::new(&shape, &strides).unwrap();
    let columns = [9., 9., 9., 9., 9., 0.5, 7., 0.75, 0.0, -0.5];
    let view_mask = [true; 5].into_iter().chain(mask).collect::<Vec<_>>();
    let view = BatchView::new(&columns, layout, Axis::Index(0))
        .unwrap()
        .with_validity(&view_mask)
        .unwrap();
    let lane = PreparedArimaFit::from_view(view, 1, order, ArimaTrend::None, skip()).unwrap();
    assert_eq!(
        evaluate(&lane).0.negative_log_likelihood,
        mask_likelihood.negative_log_likelihood
    );

    // The plan retains its byte limit for workspaces, models and forecasts.
    let bytes = ArimaWorkspace::<f64>::required_bytes(order, values.len()).unwrap();
    let limited = PreparedArimaFit::new(
        &values,
        Some(&mask),
        order,
        ArimaTrend::None,
        skip().with_max_bytes(bytes - 1),
    )
    .unwrap();
    assert_eq!(
        resource(limited.workspace()),
        ("statistics_workspace_bytes", bytes, bytes - 1)
    );
    let exact = PreparedArimaFit::new(
        &values,
        Some(&mask),
        order,
        ArimaTrend::None,
        skip().with_max_bytes(bytes),
    )
    .unwrap();
    let mut workspace = exact.workspace().unwrap();
    let model = exact.model(&parameters, &mut workspace, context).unwrap();
    let mut forecast = [ArimaForecast {
        mean: 0.,
        variance: 0.,
    }; 3];
    let mut horizon = model.workspace(3).unwrap();
    model
        .forecast_into(&mut forecast, &mut horizon, context)
        .unwrap();
    assert!((forecast[0].mean - 0.5 * -0.5).abs() < 1e-15);
    assert!((forecast[0].variance - 1.).abs() < 1e-15);
    assert_eq!(
        resource(model.workspace(1 << 20)).0,
        "statistics_workspace_bytes"
    );
    let small = PreparedArimaFit::new(
        &values,
        Some(&mask),
        order,
        ArimaTrend::None,
        skip().with_max_bytes(
            ArimaModel::<f64>::required_bytes(order, ArimaTrend::None).unwrap() - 1,
        ),
    )
    .unwrap();
    assert_eq!(
        resource(small.model(&parameters, &mut workspace, context)).0,
        "statistics_workspace_bytes"
    );
}
