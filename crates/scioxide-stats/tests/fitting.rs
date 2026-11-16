#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
use scioxide_core::*;
use scioxide_stats::*;
#[path = "support/fit_solver.rs"]
mod reference;
use reference::GridSolver;
fn identity() -> SolveIdentity {
    SolveIdentity::new(
        ProblemToken::new(1),
        scioxide_core::algorithm_id!("fit", "grid-test", 1),
        0,
        0,
    )
}
fn close<S: RealScalar>(x: S, y: f64, tol: f64) {
    let v = x.try_to_f64_exact().unwrap();
    assert!((v - y).abs() <= tol, "{v} != {y}, tolerance {tol}");
}
fn run<S: RealScalar>(
    objective: &dyn FitObjective<S, Workspace = ()>,
    bounds: &Bounds<S>,
    initial: &[S],
    intervals: usize,
) -> FitResult<S> {
    let flags = vec![false; initial.len()];
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
        .unwrap();
    scioxide_stats::fit_scoped(
        FitRequest {
            objective,
            derivatives: None,
            bounds,
            integrality: &flags,
            initial,
        },
        &mut GridSolver { intervals },
        scioxide_stats::StatisticsOptions::new().with_max_bytes(1024),
        &mut context,
    )
    .unwrap()
}
fn recovery<S: RealScalar>() {
    let s = |x| S::try_from_f64_lossy(x).unwrap();
    for (model, data, bounds, initial, expected) in [
        (
            DistributionFitModel::Normal,
            vec![-1., -1., 3., 3.],
            vec![(-1., 3.), (1., 3.)],
            vec![0., 1.],
            vec![1., 2.],
        ),
        (
            DistributionFitModel::Exponential,
            vec![1., 2., 3.],
            vec![(0.25, 0.75)],
            vec![0.25],
            vec![0.5],
        ),
        (
            DistributionFitModel::Uniform,
            vec![1., 2., 3.],
            vec![(0., 1.), (3., 4.)],
            vec![0., 4.],
            vec![1., 3.],
        ),
        (
            DistributionFitModel::Poisson,
            vec![0., 1., 2., 5.],
            vec![(0., 4.)],
            vec![1.],
            vec![2.],
        ),
    ] {
        let observations: Vec<_> = data.iter().map(|&x| FitObservation::Exact(s(x))).collect();
        let bounds = Bounds::new(
            bounds
                .iter()
                .map(|&(a, b)| (Some(s(a)), Some(s(b))))
                .collect(),
        )
        .unwrap();
        let initial: Vec<_> = initial.iter().map(|&x| s(x)).collect();
        let flags = vec![false; initial.len()];
        let problem = PreparedFit::new(
            &model,
            &observations,
            FitMethod::Likelihood,
            None,
            &bounds,
            &flags,
            scioxide_stats::StatisticsOptions::new().with_max_bytes(0),
        )
        .unwrap();
        let result = run(&problem, &bounds, &initial, 16);
        assert_eq!(result.report().termination(), StatsTermination::Converged);
        assert_eq!(
            result
                .report()
                .counts()
                .total()
                .unwrap()
                .provider(EvaluationRole::Domain),
            1
        );
        for (&x, &y) in result.parameters().unwrap().iter().zip(&expected) {
            close(x, y, 2e-5);
        }
        if matches!(model, DistributionFitModel::Uniform) {
            assert_eq!(result.report().active_bounds(), 2);
        }
    }
    // Raw moments: E[X]=1 and E[X²]=5 identify mean=1 and sigma=2.
    let observations = [FitObservation::Exact(s(-1.)), FitObservation::Exact(s(3.))];
    let bounds = Bounds::new(vec![(Some(s(0.)), Some(s(2.))), (Some(s(1.)), Some(s(3.)))]).unwrap();
    let scales = [s(1.), s(5.)];
    let problem = PreparedFit::new(
        &DistributionFitModel::Normal,
        &observations,
        FitMethod::Moments {
            orders: &[1, 2],
            scales: &scales,
        },
        None,
        &bounds,
        &[false, false],
        scioxide_stats::StatisticsOptions::new().with_max_bytes(1024),
    )
    .unwrap();
    let result = run(&problem, &bounds, &[s(0.), s(1.)], 16);
    let p = result.parameters().unwrap();
    close(p[0], 1., 2e-5);
    close(p[1], 2., 2e-5);
    close(problem.value(p, &mut ()).unwrap(), 0., 2e-6);
    let mut residual = [s(99.); 2];
    problem.residuals(p, &mut residual, &mut ()).unwrap();
    assert!(residual.iter().all(|x| x.abs() < s(1e-5)));
    // Uniform quantiles: type-7 sample quartiles of [1,2,3] match Uniform(1,3).
    let data = [
        FitObservation::Exact(s(1.)),
        FitObservation::Exact(s(2.)),
        FitObservation::Exact(s(3.)),
    ];
    let bounds = Bounds::new(vec![(Some(s(0.)), Some(s(2.))), (Some(s(2.)), Some(s(4.)))]).unwrap();
    let probabilities = [s(0.25), s(0.75)];
    let scales = [s(1.); 2];
    let problem = PreparedFit::new(
        &DistributionFitModel::Uniform,
        &data,
        FitMethod::Quantiles {
            probabilities: &probabilities,
            scales: &scales,
            convention: QuantileMethod::Linear,
        },
        None,
        &bounds,
        &[false, false],
        scioxide_stats::StatisticsOptions::new().with_max_bytes(1024),
    )
    .unwrap();
    let result = run(&problem, &bounds, &[s(0.), s(4.)], 16);
    let p = result.parameters().unwrap();
    close(p[0], 1., 2e-5);
    close(p[1], 3., 2e-5);
}
#[test]
fn parameter_recovery_both_precisions() {
    recovery::<f32>();
    recovery::<f64>();
}
fn censoring<S: RealScalar>() {
    let s = |x| S::try_from_f64_lossy(x).unwrap();
    let bounds = Bounds::new(vec![(Some(s(0.1)), Some(s(5.)))]).unwrap();
    let model = DistributionFitModel::Exponential;
    let data = [
        FitObservation::Exact(s(2.)),
        FitObservation::LeftCensored(s(1.)),
        FitObservation::RightCensored(s(3.)),
        FitObservation::IntervalCensored {
            lower: s(1.),
            upper: s(2.),
        },
    ];
    let p = PreparedFit::new(
        &model,
        &data,
        FitMethod::Likelihood,
        None,
        &bounds,
        &[false],
        scioxide_stats::StatisticsOptions::new().with_max_bytes(0),
    )
    .unwrap();
    let expected =
        2.0 - (1.0 - (-1.0_f64).exp()).ln() + 3.0 - ((-1.0_f64).exp() - (-2.0_f64).exp()).ln();
    close(p.value(&[s(1.)], &mut ()).unwrap(), expected, 2e-5);
    let data = [
        FitObservation::Exact(s(4.)),
        FitObservation::RightCensored(s(5.)),
    ];
    let p = PreparedFit::new(
        &model,
        &data,
        FitMethod::Likelihood,
        Some(FitTruncation {
            lower: Some(s(3.)),
            upper: None,
        }),
        &bounds,
        &[false],
        scioxide_stats::StatisticsOptions::new().with_max_bytes(0),
    )
    .unwrap();
    close(p.value(&[s(1.)], &mut ()).unwrap(), 3., 1e-5);
    let data = [FitObservation::IntervalCensored {
        lower: s(1000.),
        upper: s(1001.),
    }];
    let p = PreparedFit::new(
        &model,
        &data,
        FitMethod::Likelihood,
        None,
        &bounds,
        &[false],
        scioxide_stats::StatisticsOptions::new().with_max_bytes(0),
    )
    .unwrap();
    close(
        p.value(&[s(1.)], &mut ()).unwrap(),
        1000. - (1.0 - (-1.0_f64).exp()).ln(),
        2e-4,
    );
    // Discrete convention (0,2] includes mass at 1 and 2, excludes 0.
    let data = [FitObservation::IntervalCensored {
        lower: s(0.),
        upper: s(2.),
    }];
    let p = PreparedFit::new(
        &DistributionFitModel::Poisson,
        &data,
        FitMethod::Likelihood,
        None,
        &bounds,
        &[false],
        scioxide_stats::StatisticsOptions::new().with_max_bytes(0),
    )
    .unwrap();
    close(p.value(&[s(1.)], &mut ()).unwrap(), 1. - 1.5_f64.ln(), 3e-5);
    let data = [FitObservation::Exact(s(1.))];
    let p = PreparedFit::new(
        &model,
        &data,
        FitMethod::Likelihood,
        Some(FitTruncation {
            lower: None,
            upper: None,
        }),
        &bounds,
        &[false],
        scioxide_stats::StatisticsOptions::new().with_max_bytes(0),
    )
    .unwrap();
    close(p.value(&[s(1.)], &mut ()).unwrap(), 1., 1e-5);
}
#[test]
fn censoring_truncation_tail_and_count_conventions() {
    censoring::<f32>();
    censoring::<f64>();
}
#[test]
fn independent_scores_hessians_and_information() {
    let data = [-1., 0., 2., 4.];
    let observations: Vec<_> = data.iter().copied().map(FitObservation::Exact).collect();
    let bounds = Bounds::new(vec![(None, None), (Some(0.1), None)]).unwrap();
    let problem = PreparedFit::new(
        &DistributionFitModel::Normal,
        &observations,
        FitMethod::Likelihood,
        None,
        &bounds,
        &[false, false],
        scioxide_stats::StatisticsOptions::new().with_max_bytes(0),
    )
    .unwrap();
    let provider = ExactFitDerivatives {
        model: DistributionFitModel::Normal,
        samples: &data,
    };
    let p = [1.25, 2.1];
    let mut g = [0.; 2];
    let mut h = [0.; 4];
    provider.evaluate(&p, &mut g, &mut h).unwrap();
    for j in 0..2 {
        let step = 1e-4;
        let mut a = p;
        let mut b = p;
        a[j] += step;
        b[j] -= step;
        close(
            g[j],
            (problem.value(&a, &mut ()).unwrap() - problem.value(&b, &mut ()).unwrap())
                / (2. * step),
            2e-8,
        );
        let mut ga = [0.; 2];
        let mut gb = [0.; 2];
        provider.gradient(&a, &mut ga).unwrap();
        provider.gradient(&b, &mut gb).unwrap();
        for i in 0..2 {
            close(h[i * 2 + j], (ga[i] - gb[i]) / (2. * step), 5e-8);
        }
    }
    let rank = RankThreshold::new(0., 1e-10).unwrap();
    let opts = scioxide_linalg::HermitianEigenOptions::default();
    let info =
        fit_information(&p, &bounds, &[false, false], &[4., 0., 0., 2.], rank, &opts).unwrap();
    assert_eq!(info.identifiability(), FitIdentifiability::Identifiable);
    assert_eq!(info.rank(), 2);
    let cov = info.covariance().unwrap();
    close(cov[0], 0.25, 1e-12);
    close(cov[3], 0.5, 1e-12);
    let info =
        fit_information(&p, &bounds, &[false, false], &[1., 1., 1., 1.], rank, &opts).unwrap();
    assert_eq!(info.identifiability(), FitIdentifiability::RankDeficient);
    assert!(info.covariance().is_none());
    let info = fit_information(
        &p,
        &bounds,
        &[false, false],
        &[1., 0., 0., -1.],
        rank,
        &opts,
    )
    .unwrap();
    assert_eq!(info.identifiability(), FitIdentifiability::Indefinite);
    let fixed = Bounds::new(vec![(Some(p[0]), Some(p[0])), (Some(p[1]), None)]).unwrap();
    let info =
        fit_information(&p, &fixed, &[false, false], &[4., 0., 0., 2.], rank, &opts).unwrap();
    assert_eq!(info.identifiability(), FitIdentifiability::Nonregular);
}
fn transforms<S: RealScalar>() {
    let s = |x| S::try_from_f64_lossy(x).unwrap();
    for x in [0.125, 0.5, 1., 2., 8.] {
        close(
            scioxide_stats::PowerTransform::BoxCox
                .apply(s(x), s(0.))
                .unwrap(),
            x.ln(),
            2e-6,
        );
        close(
            scioxide_stats::PowerTransform::BoxCox
                .apply(s(x), s(1.))
                .unwrap(),
            x - 1.,
            2e-6,
        );
        close(
            scioxide_stats::PowerTransform::BoxCox
                .apply(s(x), s(1e-20))
                .unwrap(),
            x.ln(),
            2e-6,
        );
    }
    for x in [-3., -0.5, 0., 0.5, 3.] {
        close(
            scioxide_stats::PowerTransform::YeoJohnson
                .apply(s(x), s(1.))
                .unwrap(),
            x,
            2e-6,
        );
        if x < 0. {
            close(
                scioxide_stats::PowerTransform::YeoJohnson
                    .apply(s(x), s(2.))
                    .unwrap(),
                -(-x).ln_1p(),
                2e-6,
            );
        }
    }
    let data = [s(1.), s(2.), s(4.)];
    let expected = -1.5_f64 * (14.0_f64 / 9.0).ln();
    close(
        scioxide_stats::power_transform_llf(scioxide_stats::PowerTransform::BoxCox, s(1.), &data)
            .unwrap(),
        expected,
        2e-5,
    );
    let data = [s(-1.), s(0.), s(2.)];
    close(
        scioxide_stats::power_transform_llf(
            scioxide_stats::PowerTransform::YeoJohnson,
            s(1.),
            &data,
        )
        .unwrap(),
        expected,
        2e-5,
    );
    let mut out = [s(77.); 3];
    let mut scratch = [s(0.); 3];
    power_transform_into(
        PowerTransform::YeoJohnson,
        &data,
        s(1.),
        StatisticsOptions::new(),
        &mut out,
        &mut scratch,
        ExecutionContext::sequential(),
    )
    .unwrap();
    for (&x, &y) in out.iter().zip(&data) {
        close(x, y.try_to_f64_exact().unwrap(), 2e-6);
    }
    let old = out;
    assert!(
        power_transform_into(
            PowerTransform::BoxCox,
            &data,
            s(1.),
            StatisticsOptions::new(),
            &mut out,
            &mut scratch,
            ExecutionContext::sequential(),
        )
        .is_err()
    );
    assert_eq!(out, old);
    assert!(
        scioxide_stats::power_transform_llf(
            scioxide_stats::PowerTransform::BoxCox,
            s(1.),
            &[s(2.); 3]
        )
        .is_err()
    );
    assert!(
        scioxide_stats::PowerTransform::YeoJohnson
            .apply(S::NAN, s(0.))
            .is_err()
    );
}
#[test]
fn transformation_limits_profile_and_transactions() {
    transforms::<f32>();
    transforms::<f64>();
}
#[test]
fn jacobians_independent_data_differences() {
    for kind in [PowerTransform::BoxCox, PowerTransform::YeoJohnson] {
        for x in [-2., -0.1, 0.1, 1., 3.] {
            if kind == PowerTransform::BoxCox && x < 0. {
                continue;
            }
            for lambda in [-2., 0., 1., 2., 3.] {
                let step = 1e-5;
                let d = (kind.apply(x + step, lambda).unwrap()
                    - kind.apply(x - step, lambda).unwrap())
                    / (2. * step);
                close(kind.log_jacobian(x, lambda).unwrap(), d.ln(), 5e-8);
            }
        }
    }
}
#[test]
fn probability_plot_and_all_injected_searches() {
    let data = [1., 2., 4., 8., 16.];
    let plot = probplot_with_options(
        &data,
        &DistributionFitModel::Uniform,
        &[0., 1.],
        scioxide_stats::StatisticsOptions::new().with_max_bytes(4096),
    )
    .unwrap();
    close(plot.theoretical()[0], 1. - 0.5_f64.powf(0.2), 1e-12);
    close(plot.theoretical()[1], (2. - 0.3175) / (5. + 0.365), 1e-12);
    assert!(plot.fit().correlation.unwrap() > 0.9);
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut ctx = state
        .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
        .unwrap();
    let bounds = Bounds::new(vec![(Some(-1.), Some(1.))]).unwrap();
    let mut solver = GridSolver { intervals: 40 };
    let result = scioxide_stats::power_transform_normmax_scoped(
        scioxide_stats::PowerTransform::BoxCox,
        &data,
        PowerFitMethod::Likelihood,
        &bounds,
        0.,
        &mut solver,
        scioxide_stats::StatisticsOptions::new().with_max_bytes(4096),
        &mut ctx,
    )
    .unwrap();
    close(result.parameters().unwrap()[0], 0., 1e-12);
    let result = scioxide_stats::power_transform_normmax_scoped(
        scioxide_stats::PowerTransform::BoxCox,
        &data,
        PowerFitMethod::Pearson,
        &bounds,
        0.,
        &mut solver,
        scioxide_stats::StatisticsOptions::new().with_max_bytes(4096),
        &mut ctx,
    )
    .unwrap();
    close(result.parameters().unwrap()[0], 0., 0.051);
    let data = [-3., -1., 0., 1., 3.];
    let bounds = Bounds::new(vec![(Some(0.), Some(2.))]).unwrap();
    let result = scioxide_stats::power_transform_normmax_scoped(
        scioxide_stats::PowerTransform::YeoJohnson,
        &data,
        scioxide_stats::PowerFitMethod::Likelihood,
        &bounds,
        1.,
        &mut solver,
        scioxide_stats::StatisticsOptions::new().with_max_bytes(4096),
        &mut ctx,
    )
    .unwrap();
    close(result.parameters().unwrap()[0], 1., 0.051);
    let result = power_transform_normmax_scoped(
        PowerTransform::YeoJohnson,
        &data,
        PowerFitMethod::Pearson,
        &bounds,
        1.,
        &mut solver,
        scioxide_stats::StatisticsOptions::new().with_max_bytes(4096),
        &mut ctx,
    )
    .unwrap();
    close(result.parameters().unwrap()[0], 1., 0.051);
    for kind in [PowerTransform::BoxCox, PowerTransform::YeoJohnson] {
        assert!(
            power_transform_normmax_scoped(
                kind,
                &[1., 2., 4.],
                PowerFitMethod::Likelihood,
                &bounds,
                1.,
                &mut solver,
                scioxide_stats::StatisticsOptions::new().with_max_bytes(15),
                &mut ctx
            )
            .is_err()
        );
        let result = power_transform_normmax_scoped(
            kind,
            &[1., 2., 4.],
            PowerFitMethod::Likelihood,
            &bounds,
            1.,
            &mut solver,
            scioxide_stats::StatisticsOptions::new().with_max_bytes(16),
            &mut ctx,
        )
        .unwrap();
        assert_eq!(
            result.parameters().is_some(),
            result.report().objective().is_some()
        );
    }
    // A one-parameter quantile family makes the PPCC maximum identifiable.
    struct Shape;
    impl FitModel<f64> for Shape {
        fn parameter_count(&self) -> usize {
            1
        }
        fn validate(&self, p: &[f64]) -> Result<(), SciOxideError> {
            if p.len() == 1 && p[0] > 0. {
                Ok(())
            } else {
                Err(SciOxideError::not_implemented("test shape"))
            }
        }
        fn log_observation(&self, _: &[f64], _: FitObservation<f64>) -> Result<f64, SciOxideError> {
            Err(SciOxideError::not_implemented("quantile-only family"))
        }
        fn quantile(&self, p: &[f64], q: f64) -> Result<f64, SciOxideError> {
            Ok(q.powf(p[0]))
        }
    }
    let uniform = probplot_with_options(
        &[0., 1., 2., 3., 4.],
        &DistributionFitModel::Uniform,
        &[0., 1.],
        scioxide_stats::StatisticsOptions::new().with_max_bytes(4096),
    )
    .unwrap();
    let data: Vec<_> = uniform.theoretical().iter().map(|q| q * q).collect();
    let bounds = Bounds::new(vec![(Some(1.), Some(3.))]).unwrap();
    let problem = PreparedPpcc::new(
        &data,
        &Shape,
        &bounds,
        &[false],
        scioxide_stats::StatisticsOptions::new().with_max_bytes(4096),
    )
    .unwrap();
    let result = ppcc_max_scoped(
        &problem,
        &[1.],
        &mut solver,
        scioxide_stats::StatisticsOptions::new().with_max_bytes(4096),
        &mut ctx,
    )
    .unwrap();
    close(result.parameters().unwrap()[0], 2., 1e-12);
}
#[test]
fn capability_budget_and_bad_candidate_boundaries() {
    struct Candidate {
        bad: bool,
        calls: usize,
    }
    impl FitSolver<f64> for Candidate {
        fn capabilities(&self) -> FitSolverCapabilities {
            FitSolverCapabilities {
                bounds: true,
                ..Default::default()
            }
        }
        fn solve(
            &mut self,
            _: FitRequest<'_, f64>,
            out: &mut [f64],
            _: &mut (),
            _: &mut SolveContext<'_, '_>,
        ) -> Result<StatsTermination, SciOxideError> {
            self.calls += 1;
            out[0] = if self.bad { f64::NAN } else { 1. };
            Ok(StatsTermination::Converged)
        }
    }
    let data = [FitObservation::Exact(1.)];
    let bounds = Bounds::new(vec![(Some(0.1), Some(2.))]).unwrap();
    let problem = PreparedFit::new(
        &DistributionFitModel::Exponential,
        &data,
        FitMethod::Likelihood,
        None,
        &bounds,
        &[false],
        scioxide_stats::StatisticsOptions::new().with_max_bytes(0),
    )
    .unwrap();
    let request = FitRequest {
        objective: &problem,
        derivatives: None,
        bounds: &bounds,
        integrality: &[false],
        initial: &[1.],
    };
    let mut workspace = FitWorkspace::new(
        1,
        scioxide_stats::StatisticsOptions::new().with_max_bytes(8),
    )
    .unwrap();
    let mut output = [99.];
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, Budget::ONE);
    let mut ctx = state
        .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
        .unwrap();
    assert!(
        ctx.record_provider(EvaluationRole::Objective)
            .unwrap()
            .is_granted()
    );
    let mut solver = Candidate {
        bad: false,
        calls: 0,
    };
    let report = fit_into(request, &mut solver, &mut output, &mut workspace, &mut ctx).unwrap();
    assert_eq!(report.termination(), StatsTermination::BudgetExhausted);
    assert!(report.objective().is_none());
    assert_eq!(output, [99.]);
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut ctx = state
        .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
        .unwrap();
    solver.bad = true;
    assert!(fit_into(request, &mut solver, &mut output, &mut workspace, &mut ctx).is_err());
    assert_eq!(output, [99.]);
    let integer = [true];
    let request = FitRequest {
        integrality: &integer,
        ..request
    };
    let before = solver.calls;
    assert!(
        matches ! (& (fit_into (request , & mut solver , & mut output , & mut workspace , & mut ctx)) , Err (__scioxide_error) if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: NotImplemented { .. }))
    );
    assert_eq!(solver.calls, before);
    assert!(
        FitWorkspace::<f64>::new(
            2,
            scioxide_stats::StatisticsOptions::new().with_max_bytes(15)
        )
        .is_err()
    );
    assert!(
        PreparedFit::new(
            &DistributionFitModel::Exponential,
            &[],
            FitMethod::Likelihood,
            None,
            &bounds,
            &[false],
            scioxide_stats::StatisticsOptions::new().with_max_bytes(0)
        )
        .is_err()
    );
    let censored = [FitObservation::RightCensored(1.)];
    assert!(
        PreparedFit::new(
            &DistributionFitModel::Exponential,
            &censored,
            FitMethod::Moments {
                orders: &[1],
                scales: &[1.]
            },
            None,
            &bounds,
            &[false],
            scioxide_stats::StatisticsOptions::new().with_max_bytes(1024)
        )
        .is_err()
    );
}

#[test]
fn high_precision_defining_equation_fixtures() {
    for line in include_str!("data/fitting.tsv")
        .lines()
        .filter(|line| !line.starts_with('#'))
    {
        let fields: Vec<_> = line.split_whitespace().collect();
        let kind = if fields[0].starts_with("boxcox") {
            PowerTransform::BoxCox
        } else {
            PowerTransform::YeoJohnson
        };
        let x: f64 = fields[1].parse().unwrap();
        let lambda: f64 = fields[2].parse().unwrap();
        let expected: f64 = fields[3].parse().unwrap();
        let jac: f64 = fields[4].parse().unwrap();
        if fields[0].ends_with("llf") {
            let data: &[f64] = if kind == PowerTransform::BoxCox {
                &[0.1, 0.7, 2., 10., 31.]
            } else {
                &[-30., -0.1, 0., 0.1, 2., 10.]
            };
            close(
                power_transform_llf(kind, lambda, data).unwrap(),
                expected,
                2e-11 * expected.abs().max(1.),
            );
            let single: Vec<_> = data.iter().map(|&x| x as f32).collect();
            close(
                power_transform_llf(kind, lambda as f32, &single).unwrap(),
                expected,
                5e-5 * expected.abs().max(1.),
            );
        } else {
            close(
                kind.apply(x, lambda).unwrap(),
                expected,
                2e-12 * expected.abs().max(1.),
            );
            close(
                kind.log_jacobian(x, lambda).unwrap(),
                jac,
                2e-12 * jac.abs().max(1.),
            );
            close(
                kind.apply(x as f32, lambda as f32).unwrap(),
                expected,
                4e-6 * expected.abs().max(1.),
            );
            close(
                kind.log_jacobian(x as f32, lambda as f32).unwrap(),
                jac,
                4e-6 * jac.abs().max(1.),
            );
        }
    }
    // Intermediate exp overflow must not discard a representable quotient.
    let x = (710.0_f64 / 10.0).exp();
    close(
        scioxide_stats::PowerTransform::BoxCox
            .apply(x, 10.)
            .unwrap(),
        (710.0_f64 - 10.0_f64.ln()).exp() - 0.1,
        2e-13 * (710.0_f64 - 10.0_f64.ln()).exp(),
    );
}
#[test]
fn generic_law_nonidentifiability_and_zero_rate_boundary() {
    let family =
        LawFitModel::new(2, |p: &[f64]| scioxide_random::Normal::new(p[0] + p[1], 1.)).unwrap();
    let data = [FitObservation::Exact(1.), FitObservation::Exact(3.)];
    let bounds = Bounds::new(vec![(Some(0.), Some(2.)); 2]).unwrap();
    let problem = PreparedFit::new(
        &family,
        &data,
        FitMethod::Likelihood,
        None,
        &bounds,
        &[false, false],
        scioxide_stats::StatisticsOptions::new().with_max_bytes(0),
    )
    .unwrap();
    close(
        problem.value(&[0., 2.], &mut ()).unwrap(),
        problem.value(&[1., 1.], &mut ()).unwrap(),
        1e-12,
    );
    let info = fit_information(
        &[1., 1.],
        &bounds,
        &[false, false],
        &[2., 2., 2., 2.],
        RankThreshold::new(0., 1e-10).unwrap(),
        &Default::default(),
    )
    .unwrap();
    assert_eq!(info.identifiability(), FitIdentifiability::RankDeficient);
    let bounds = Bounds::new(vec![(Some(0.), Some(2.))]).unwrap();
    let data = [FitObservation::Exact(0.); 3];
    let problem = PreparedFit::new(
        &DistributionFitModel::Poisson,
        &data,
        FitMethod::Likelihood,
        None,
        &bounds,
        &[false],
        scioxide_stats::StatisticsOptions::new().with_max_bytes(0),
    )
    .unwrap();
    let result = run(&problem, &bounds, &[1.], 16);
    assert_eq!(result.parameters().unwrap(), vec![0.]);
    assert_eq!(result.report().active_bounds(), 1);
    for model in [
        DistributionFitModel::Exponential,
        DistributionFitModel::Poisson,
    ] {
        let provider = ExactFitDerivatives {
            model,
            samples: &[0.5_f32, 1., 2.],
        };
        let mut gradient = [0.];
        let mut hessian = [0.];
        if matches!(model, DistributionFitModel::Exponential) {
            provider
                .evaluate(&[2.], &mut gradient, &mut hessian)
                .unwrap();
            close(gradient[0], 2., 1e-6);
            close(hessian[0], 0.75, 1e-6);
        } else {
            assert!(
                provider
                    .evaluate(&[2.], &mut gradient, &mut hessian)
                    .is_err()
            );
        }
    }
}
#[cfg(feature = "std")]
#[test]
fn executor_batches_and_cancellation_are_transactional() {
    use core::num::NonZeroUsize;
    let workers = NonZeroUsize::new(3).unwrap();
    let pool = ThreadExecutor::new(workers);
    let options =
        StatisticsOptions::new().with_execution_policy(ExecutionPolicy::Parallel(workers));
    let samples: Vec<_> = (0..96).map(|i| (i as f64 - 40.) / 10.).collect();
    let mut serial = vec![0.; samples.len()];
    let mut scratch = serial.clone();
    let mut parallel = serial.clone();
    power_transform_into(
        PowerTransform::YeoJohnson,
        &samples,
        0.7,
        Default::default(),
        &mut serial,
        &mut scratch,
        ExecutionContext::sequential(),
    )
    .unwrap();
    power_transform_into(
        PowerTransform::YeoJohnson,
        &samples,
        0.7,
        options,
        &mut parallel,
        &mut scratch,
        ExecutionContext::new(&pool),
    )
    .unwrap();
    assert_eq!(serial, parallel);
    let data = [FitObservation::Exact(1.), FitObservation::Exact(2.)];
    let bounds = Bounds::new(vec![(Some(0.1), Some(20.))]).unwrap();
    let problem = PreparedFit::new(
        &DistributionFitModel::Exponential,
        &data,
        FitMethod::Likelihood,
        None,
        &bounds,
        &[false],
        scioxide_stats::StatisticsOptions::new().with_max_bytes(0),
    )
    .unwrap();
    let parameters: Vec<_> = (1..97).map(|i| i as f64 / 10.).collect();
    fit_objectives_into(
        &problem,
        &parameters,
        Default::default(),
        &mut serial,
        &mut scratch,
        &mut [(); 64],
        ExecutionContext::sequential(),
    )
    .unwrap();
    fit_objectives_into(
        &problem,
        &parameters,
        options,
        &mut parallel,
        &mut scratch,
        &mut [(); 64],
        ExecutionContext::new(&pool),
    )
    .unwrap();
    assert_eq!(serial, parallel);
    let token = CancellationToken::new();
    token.cancel();
    assert!(
        fit_objectives_into(
            &problem,
            &parameters,
            options,
            &mut parallel,
            &mut scratch,
            &mut [(); 64],
            ExecutionContext::new(&pool).with_cancellation(&token),
        )
        .is_err()
    );
    assert_eq!(serial, parallel);
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut context = state
        .begin_solve(
            ExecutionContext::sequential().with_cancellation(&token),
            identity(),
            &mut budgets,
        )
        .unwrap();
    let request = FitRequest {
        objective: &problem,
        derivatives: None,
        bounds: &bounds,
        integrality: &[false],
        initial: &[1.],
    };
    let result = scioxide_stats::fit_scoped(
        request,
        &mut GridSolver { intervals: 16 },
        scioxide_stats::StatisticsOptions::new().with_max_bytes(1024),
        &mut context,
    )
    .unwrap();
    assert_eq!(
        result.report().termination(),
        StatsTermination::Stopped(scioxide_core::ExecutionStopReason::Cancelled)
    );
    assert!(result.parameters().is_none());
    assert_eq!(
        result
            .report()
            .counts()
            .total()
            .unwrap()
            .provider(EvaluationRole::Objective),
        0
    );
}

#[test]
fn single_precision_information_and_regular_count_score() {
    let provider = ExactFitDerivatives {
        model: DistributionFitModel::Normal,
        samples: &[-1.0_f32, -1., 3., 3.],
    };
    let mut gradient = [0.0; 2];
    let mut hessian = [0.0; 4];
    provider
        .evaluate(&[1., 2.], &mut gradient, &mut hessian)
        .unwrap();
    assert_eq!(gradient, [0., 0.]);
    assert_eq!(hessian, [1., 0., 0., 2.]);
    let bounds = Bounds::new(vec![(None, None), (Some(0.), None)]).unwrap();
    let information = fit_information(
        &[1., 2.],
        &bounds,
        &[false, false],
        &hessian,
        RankThreshold::new(0., 1e-5).unwrap(),
        &Default::default(),
    )
    .unwrap();
    assert_eq!(
        information.identifiability(),
        FitIdentifiability::Identifiable
    );
    assert_eq!(information.covariance().unwrap(), vec![1., 0., 0., 0.5]);
    let poisson = ExactFitDerivatives {
        model: DistributionFitModel::Poisson,
        samples: &[0.0_f32, 1., 2., 5.],
    };
    let mut gradient = [0.0];
    let mut hessian = [0.0];
    poisson
        .evaluate(&[2.], &mut gradient, &mut hessian)
        .unwrap();
    assert_eq!(gradient, [0.]);
    assert_eq!(hessian, [2.]);
}

#[test]
fn profile_variance_preserves_shifted_and_extended_range_samples() {
    close(
        scioxide_stats::power_transform_llf(
            scioxide_stats::PowerTransform::BoxCox,
            1.,
            &[1e10, 1e10 + 1., 1e10 + 2.],
        )
        .unwrap(),
        -1.5 * (2.0_f64 / 3.0).ln(),
        2e-12,
    );
    close(
        scioxide_stats::power_transform_llf(
            scioxide_stats::PowerTransform::BoxCox,
            1.0_f32,
            &[1e6, 1e6 + 1., 1e6 + 2.],
        )
        .unwrap(),
        -1.5 * (2.0_f64 / 3.0).ln(),
        2e-6,
    );
    close(
        scioxide_stats::power_transform_llf(
            scioxide_stats::PowerTransform::BoxCox,
            1.,
            &[1e200, 2e200, 3e200],
        )
        .unwrap(),
        -1.5 * ((2.0_f64 / 3.0).ln() + 400.0 * 10.0_f64.ln()),
        2e-11,
    );
    close(
        scioxide_stats::power_transform_llf(
            scioxide_stats::PowerTransform::YeoJohnson,
            1.,
            &[-1e-200, 0., 1e-200],
        )
        .unwrap(),
        -1.5 * ((2.0_f64 / 3.0).ln() - 400.0 * 10.0_f64.ln()),
        2e-11,
    );
}
