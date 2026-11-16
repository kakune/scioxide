#![allow(
    clippy::unwrap_used,
    clippy::panic,
    reason = "checked test fixtures and expected reports"
)]

use scioxide_core::{
    EvaluationRole, ExecutionContext, ProblemToken, RankThreshold, SciOxideError, SolveBudgets,
    SolveContext, SolveIdentity, SolveState,
};
use scioxide_linalg::{HermitianEigenOptions, Matrix, MatrixView};
use scioxide_stats::{
    ConfidenceProbability, CoxInferenceOptions, CoxTies, FitDerivatives, FitObjective, FitRequest,
    FitSolver, FitSolverCapabilities, PreparedCox, StatsTermination, SurvivalObservation,
    SurvivalOptions, cox_fit_scoped, kaplan_meier_into, kaplan_meier_with_options,
    logrank_test_with_options, nelson_aalen_into, nelson_aalen_with_options,
};

fn row(entry: Option<f64>, time: f64, event: bool) -> SurvivalObservation<f64> {
    SurvivalObservation::new(
        entry,
        time,
        if event {
            scioxide_stats::SurvivalOutcome::Event
        } else {
            scioxide_stats::SurvivalOutcome::Censored
        },
    )
    .unwrap()
}

#[test]
fn exact_delayed_entry_risk_sets_and_estimators() -> Result<(), SciOxideError> {
    let data = [
        row(None, 2.0, true),
        row(None, 2.0, false),
        row(Some(1.0), 3.0, true),
        row(Some(2.5), 3.0, true),
    ];
    let km = kaplan_meier_with_options(
        &data,
        ConfidenceProbability::default(),
        scioxide_stats::SurvivalOptions::new().with_max_bytes(4096),
    )?;
    let na = nelson_aalen_with_options(
        &data,
        ConfidenceProbability::default(),
        scioxide_stats::SurvivalOptions::new().with_max_bytes(4096),
    )?;
    assert_eq!(km.len(), 2);
    assert_eq!((km[0].at_risk, km[0].events, km[0].censored), (3, 1, 1));
    assert!((km[0].survival - 2.0 / 3.0).abs() < 1e-14);
    assert!(km[0].interval.is_some());
    assert_eq!((km[1].at_risk, km[1].events), (2, 2));
    assert_eq!(km[1].survival, 0.0);
    assert!(km[1].interval.is_none());
    assert!((na[0].cumulative_hazard - 1.0 / 3.0).abs() < 1e-14);
    assert!((na[1].cumulative_hazard - 4.0 / 3.0).abs() < 1e-14);
    assert!((na[1].variance - (1.0 / 9.0 + 0.5)).abs() < 1e-14);
    let mut km_output = [km[0]; 2];
    let mut na_output = [na[0]; 2];
    assert_eq!(
        kaplan_meier_into(
            &data,
            ConfidenceProbability::default(),
            scioxide_stats::SurvivalOptions::new().with_max_bytes(4096),
            &mut km_output,
            &mut curve_workspace(&data, scioxide_stats::SurvivalEstimator::KaplanMeier),
            scioxide_core::ExecutionContext::sequential()
        )?,
        2
    );
    assert_eq!(
        nelson_aalen_into(
            &data,
            ConfidenceProbability::default(),
            scioxide_stats::SurvivalOptions::new().with_max_bytes(4096),
            &mut na_output,
            &mut curve_workspace(&data, scioxide_stats::SurvivalEstimator::NelsonAalen),
            scioxide_core::ExecutionContext::sequential()
        )?,
        2
    );
    assert_eq!(km_output[1].survival, km[1].survival);
    assert_eq!(na_output[1].cumulative_hazard, na[1].cumulative_hazard);
    let mut short = [km[0]];
    assert!(
        kaplan_meier_into(
            &data,
            ConfidenceProbability::default(),
            scioxide_stats::SurvivalOptions::new().with_max_bytes(4096),
            &mut short,
            &mut curve_workspace(&data, scioxide_stats::SurvivalEstimator::KaplanMeier),
            scioxide_core::ExecutionContext::sequential()
        )
        .is_err()
    );
    assert_eq!(short[0].survival, km[0].survival);
    Ok(())
}

#[test]
fn entry_at_event_time_joins_only_later_risk_sets() -> Result<(), SciOxideError> {
    let data = [row(None, 2., true), row(Some(2.), 3., true)];
    let curve = kaplan_meier_with_options(
        &data,
        ConfidenceProbability::default(),
        scioxide_stats::SurvivalOptions::new().with_max_bytes(1024),
    )?;
    assert_eq!(curve.iter().map(|p| p.at_risk).collect::<Vec<_>>(), [1, 1]);
    assert_eq!(curve[0].survival, 0.);
    assert_eq!(curve[1].survival, 0.);
    Ok(())
}

#[test]
fn exact_logrank_counts_with_ties() -> Result<(), SciOxideError> {
    let first = [row(None, 2.0, true), row(Some(1.0), 3.0, true)];
    let second = [row(None, 2.0, false), row(Some(2.5), 3.0, true)];
    let result =
        logrank_test_with_options(&first, &second, SurvivalOptions::new().with_max_bytes(4096))?;
    assert_eq!(result.observed(), [2, 1]);
    assert!((result.expected()[0] - 5.0 / 3.0).abs() < 1e-14);
    assert!((result.variance() - 2.0 / 9.0).abs() < 1e-14);
    assert!((result.statistic() - 0.5).abs() < 1e-14);
    Ok(())
}

#[test]
fn all_censored_and_invalid_inputs() -> Result<(), SciOxideError> {
    let censored = [row(None, 1.0, false), row(Some(0.5), 2.0, false)];
    assert!(
        kaplan_meier_with_options(
            &censored,
            ConfidenceProbability::default(),
            scioxide_stats::SurvivalOptions::new().with_max_bytes(4096)
        )?
        .is_empty()
    );
    assert!(
        nelson_aalen_with_options(
            &censored,
            ConfidenceProbability::default(),
            scioxide_stats::SurvivalOptions::new().with_max_bytes(4096)
        )?
        .is_empty()
    );
    assert!(
        logrank_test_with_options(
            &censored,
            &censored,
            SurvivalOptions::new().with_max_bytes(4096)
        )
        .is_err()
    );
    assert!(
        SurvivalObservation::new(Some(1.0), 1.0, scioxide_stats::SurvivalOutcome::Event).is_err()
    );
    assert!(
        SurvivalObservation::new(None, f64::NAN, scioxide_stats::SurvivalOutcome::Event).is_err()
    );
    assert!(
        kaplan_meier_with_options(
            &censored,
            ConfidenceProbability::default(),
            scioxide_stats::SurvivalOptions::new().with_max_bytes(0)
        )
        .is_ok()
    );
    assert!(
        kaplan_meier_with_options(
            &[row(None, 1.0, true)],
            ConfidenceProbability::default(),
            scioxide_stats::SurvivalOptions::new().with_max_bytes(0)
        )
        .is_err()
    );
    assert!(
        logrank_test_with_options(
            &[row(None, 1., true)],
            &censored,
            SurvivalOptions::new().with_max_bytes(0)
        )
        .is_err()
    );
    Ok(())
}

#[test]
fn cox_tie_likelihood_score_and_information() -> Result<(), SciOxideError> {
    let data = [
        row(None, 1., true),
        row(None, 1., true),
        row(None, 2., true),
    ];
    let design = MatrixView::from_slice(3, 1, &[0., 1., 2.])?;
    let efron = PreparedCox::new(
        &data,
        design,
        CoxTies::Efron,
        SurvivalOptions::new().with_max_bytes(1024),
    )?;
    let breslow = PreparedCox::new(
        &data,
        design,
        CoxTies::Breslow,
        SurvivalOptions::new().with_max_bytes(1024),
    )?;
    let mut gradient = [0.];
    let mut hessian = [0.];
    assert!((efron.value(&[0.], &mut ())? - 6.0_f64.ln()).abs() < 1e-14);
    assert!((breslow.value(&[0.], &mut ())? - 2.0 * 3.0_f64.ln()).abs() < 1e-14);
    FitDerivatives::gradient(&efron, &[0.], &mut gradient)?;
    FitDerivatives::hessian(&efron, &[0.], &mut hessian)?;
    assert!((gradient[0] - 1.25).abs() < 1e-14);
    assert!((hessian[0] - 65.0 / 48.0).abs() < 1e-14);
    FitDerivatives::gradient(&breslow, &[0.], &mut gradient)?;
    FitDerivatives::hessian(&breslow, &[0.], &mut hessian)?;
    assert!((gradient[0] - 1.0).abs() < 1e-14);
    assert!((hessian[0] - 4.0 / 3.0).abs() < 1e-14);
    let owned = Matrix::from_vec(3, 1, vec![0., 1., 2.])?;
    let owned_cox = PreparedCox::new(
        &data,
        owned.view(),
        CoxTies::Efron,
        SurvivalOptions::new().with_max_bytes(1024),
    )?;
    let strided_data = [0., 99., 1., 99., 2.];
    let strided = MatrixView::from_slice_with_strides(3, 1, &strided_data, 2, 1)?;
    let strided_cox = PreparedCox::new(
        &data,
        strided,
        CoxTies::Efron,
        SurvivalOptions::new().with_max_bytes(1024),
    )?;
    assert_eq!(
        owned_cox.value(&[0.], &mut ())?,
        efron.value(&[0.], &mut ())?
    );
    assert_eq!(
        strided_cox.value(&[0.], &mut ())?,
        efron.value(&[0.], &mut ())?
    );
    Ok(())
}

#[test]
fn cox_information_survives_covariate_translation() -> Result<(), SciOxideError> {
    let data = [
        row(None, 1.0, true),
        row(None, 2.0, true),
        row(None, 3.0, true),
    ];
    for ties in [CoxTies::Efron, CoxTies::Breslow] {
        let shifted = [
            2.0_f64.powi(30),
            2.0_f64.powi(30) + 1.0,
            2.0_f64.powi(30) + 2.0,
        ];
        let model = PreparedCox::new(
            &data,
            MatrixView::from_slice(3, 1, &shifted)?,
            ties,
            SurvivalOptions::new().with_max_bytes(1024),
        )?;
        let mut information = [0.0];
        let mut gradient = [0.0];
        model.hessian(&[0.0], &mut information)?;
        model.gradient(&[0.0], &mut gradient)?;
        assert!((information[0] - 11.0 / 12.0).abs() < 2e-14);
        assert!((gradient[0] - 1.5).abs() < 2e-14);
        assert!((model.value(&[0.0], &mut ())? - 6.0_f64.ln()).abs() < 2e-14);
    }
    Ok(())
}

fn rational_cox_derivatives<S: scioxide_core::RealScalar>(
    offset: f64,
    exponents: &[i32],
    tolerance: f64,
) -> Result<(), SciOxideError> {
    let s = S::try_from_f64_lossy;
    let data = [
        SurvivalObservation::new(None, s(1.0)?, scioxide_stats::SurvivalOutcome::Event)?,
        SurvivalObservation::new(None, s(1.0)?, scioxide_stats::SurvivalOutcome::Event)?,
        SurvivalObservation::new(
            Some(s(0.5)?),
            s(2.0)?,
            scioxide_stats::SurvivalOutcome::Event,
        )?,
        SurvivalObservation::new(
            Some(s(1.5)?),
            s(3.0)?,
            scioxide_stats::SurvivalOutcome::Event,
        )?,
    ];
    let canonical = [[0.0, 1.0], [1.0, -1.0], [2.0, 2.0], [3.0, 0.0]];
    // At beta=(ln 2, ln 2), risk weights are exactly (2,1,16,8).
    // These rational derivatives use the independent pair identity
    // sum_{i<j} w_i*w_j*(x_i-x_j)*(x_i-x_j)^T / sum(w)^2.
    for ties in [CoxTies::Efron, CoxTies::Breslow] {
        let (value, gradient, information) = if ties == CoxTies::Efron {
            (
                5.518957787644128,
                1168.0 / 399.0,
                [
                    687262.0 / 796005.0,
                    27316.0 / 796005.0,
                    27316.0 / 796005.0,
                    1347208.0 / 796005.0,
                ],
            )
        } else {
            (
                5.6011958858811,
                160.0 / 57.0,
                [
                    3350.0 / 3249.0,
                    500.0 / 3249.0,
                    500.0 / 3249.0,
                    6200.0 / 3249.0,
                ],
            )
        };
        for &exponent in exponents {
            let scales = [2.0_f64.powi(exponent), -2.0_f64.powi(exponent)];
            let mut design = Vec::new();
            for x in canonical {
                design.extend([
                    s((x[0] + offset) * scales[0])?,
                    s((x[1] - offset) * scales[1])?,
                ]);
            }
            let beta = [
                s(core::f64::consts::LN_2 / scales[0])?,
                s(core::f64::consts::LN_2 / scales[1])?,
            ];
            let model = PreparedCox::new(
                &data,
                MatrixView::from_slice(4, 2, &design)?,
                ties,
                SurvivalOptions::new().with_max_bytes(1024),
            )?;
            assert!((model.value(&beta, &mut ())?.into_f64() - value).abs() < tolerance);
            let mut g = [S::ZERO; 2];
            let mut h = [S::ZERO; 4];
            model.gradient(&beta, &mut g)?;
            model.hessian(&beta, &mut h)?;
            for (col, g) in g.into_iter().enumerate() {
                assert!((g.into_f64() / scales[col] - gradient).abs() < tolerance);
            }
            for (index, h) in h.into_iter().enumerate() {
                assert!(
                    (h.into_f64() / scales[index / 2] / scales[index % 2] - information[index])
                        .abs()
                        < tolerance
                );
            }
        }
    }
    Ok(())
}

#[test]
fn cox_ties_and_delayed_entry_match_independent_rational_derivatives() -> Result<(), SciOxideError>
{
    rational_cox_derivatives::<f64>(2.0_f64.powi(30), &[0, 500, -500], 2e-13)?;
    rational_cox_derivatives::<f32>(2.0_f64.powi(20), &[0, 50, -50], 3e-6)
}

#[test]
fn constant_covariates_remain_singular_even_with_overflowing_raw_predictors()
-> Result<(), SciOxideError> {
    let data = [
        row(None, 1.0, true),
        row(None, 2.0, true),
        row(None, 3.0, true),
    ];
    let x = [1e308; 3];
    let model = PreparedCox::new(
        &data,
        MatrixView::from_slice(3, 1, &x)?,
        CoxTies::Efron,
        SurvivalOptions::new().with_max_bytes(1024),
    )?;
    let mut h = [1.0];
    let mut g = [1.0];
    model.gradient(&[1e308], &mut g)?;
    model.hessian(&[1e308], &mut h)?;
    assert_eq!(g, [0.0]);
    assert_eq!(h, [0.0]);
    assert!((model.value(&[1e308], &mut ())? - 6.0_f64.ln()).abs() < 2e-14);
    Ok(())
}

#[test]
fn cox_projects_only_requested_quantities_and_retains_tiny_risk_weights()
-> Result<(), SciOxideError> {
    let data = [row(None, 1.0, true), row(None, 2.0, true)];
    let endpoints = [-f64::MAX, f64::MAX];
    let model = PreparedCox::new(
        &data,
        MatrixView::from_slice(2, 1, &endpoints)?,
        CoxTies::Efron,
        SurvivalOptions::new().with_max_bytes(1024),
    )?;
    let mut gradient = [0.0];
    let mut hessian = [0.0];
    assert!((model.value(&[0.0], &mut ())? - 2.0_f64.ln()).abs() < 1e-14);
    model.gradient(&[0.0], &mut gradient)?;
    assert!((gradient[0] / f64::MAX - 1.0).abs() < 1e-14);
    assert!(model.hessian(&[0.0], &mut hessian).is_err());

    // exp(-1000) underflows, but its product with the squared contrast does not.
    let x = [-1e250, 0.0];
    let beta = [1e-247];
    let model = PreparedCox::new(
        &data,
        MatrixView::from_slice(2, 1, &x)?,
        CoxTies::Efron,
        SurvivalOptions::new().with_max_bytes(1024),
    )?;
    model.hessian(&beta, &mut hessian)?;
    let expected = (2.0 * x[0].abs().ln() + x[0] * beta[0]).exp();
    assert!((hessian[0] / expected - 1.0).abs() < 2e-12);
    Ok(())
}

#[test]
fn cox_delayed_entry_and_domains() -> Result<(), SciOxideError> {
    let data = [
        row(None, 1., true),
        row(None, 1., true),
        row(Some(1.5), 2., true),
    ];
    let design = MatrixView::from_slice(3, 1, &[0., 1., 2.])?;
    let cox = PreparedCox::new(
        &data,
        design,
        CoxTies::Efron,
        SurvivalOptions::new().with_max_bytes(1024),
    )?;
    assert!((cox.value(&[0.], &mut ())? - 2.0_f64.ln()).abs() < 1e-14);
    assert!(cox.value(&[], &mut ()).is_err());
    assert!(cox.value(&[f64::INFINITY], &mut ()).is_err());
    assert!(
        PreparedCox::new(
            &data,
            design,
            CoxTies::Efron,
            SurvivalOptions::new().with_max_bytes(0)
        )
        .is_err()
    );
    let all_censored = [row(None, 1., false)];
    assert!(
        PreparedCox::new(
            &all_censored,
            MatrixView::from_slice(1, 1, &[1.])?,
            CoxTies::Efron,
            SurvivalOptions::new().with_max_bytes(1024)
        )
        .is_err()
    );
    Ok(())
}

struct FixedStationarySolver;

impl FitSolver<f64> for FixedStationarySolver {
    fn capabilities(&self) -> FitSolverCapabilities {
        FitSolverCapabilities::default()
    }

    fn solve(
        &mut self,
        request: FitRequest<'_, f64>,
        candidate: &mut [f64],
        _: &mut (),
        context: &mut SolveContext<'_, '_>,
    ) -> Result<StatsTermination, SciOxideError> {
        if !context.record_iteration()?.is_granted() {
            return Ok(StatsTermination::BudgetExhausted);
        }
        if !context
            .record_provider(EvaluationRole::Gradient)?
            .is_granted()
        {
            return Ok(StatsTermination::BudgetExhausted);
        }
        let mut gradient = [0.0];
        request
            .derivatives
            .unwrap()
            .gradient(candidate, &mut gradient)?;
        if gradient[0].abs() > 1e-12 {
            return Ok(StatsTermination::NotConverged);
        }
        Ok(StatsTermination::Converged)
    }
}

#[test]
fn cox_injected_fit_information_baseline_and_residual() -> Result<(), SciOxideError> {
    let data = [
        row(None, 2., false),
        row(None, 1., true),
        row(None, 2., false),
    ];
    let design = MatrixView::from_slice(3, 1, &[-1., 0., 1.])?;
    let problem = PreparedCox::new(
        &data,
        design,
        CoxTies::Efron,
        SurvivalOptions::new().with_max_bytes(1024),
    )?;
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let identity = SolveIdentity::new(
        ProblemToken::new(555),
        scioxide_core::algorithm_id!("survival-test", "stationary", 1),
        0,
        0,
    );
    let mut context = state.begin_solve(ExecutionContext::sequential(), identity, &mut budgets)?;
    let options = CoxInferenceOptions::new()
        .with_confidence(ConfidenceProbability::default())
        .with_rank_tolerance(RankThreshold::new(0., 1e-10)?)
        .with_eigen_options(HermitianEigenOptions::default())
        .with_max_fit_bytes(1024)
        .with_max_report_bytes(4096);
    let result = cox_fit_scoped(
        &problem,
        &[0.],
        &mut FixedStationarySolver,
        options,
        &mut context,
    )?;
    assert_eq!(result.fit().parameters(), Some(&[0.][..]));
    let report = result.report().unwrap();
    assert_eq!(report.information().rank(), 1);
    assert!((report.information().covariance().as_ref().unwrap()[0] - 1.5).abs() < 1e-12);
    assert_eq!(report.hazard_ratios(), [1.]);
    assert_eq!(
        report
            .schoenfeld_residuals()
            .map(|row| row.observation())
            .collect::<Vec<_>>(),
        [1]
    );
    assert!(report.schoenfeld_residuals().next().unwrap().values()[0].abs() < 1e-14);
    assert!((report.baseline()[0].cumulative_hazard - 1.0 / 3.0).abs() < 1e-14);
    let interval = report.coefficient_intervals().unwrap()[0];
    assert!(interval.lower() < 0. && interval.upper() > 0.);
    assert!(
        cox_fit_scoped(
            &problem,
            &[0.],
            &mut FixedStationarySolver,
            options.with_max_fit_bytes(0),
            &mut context
        )
        .is_err()
    );
    Ok(())
}

#[test]
fn cox_derivatives_match_independent_differences() -> Result<(), SciOxideError> {
    let data = [
        row(None, 1., true),
        row(None, 2., true),
        row(Some(0.5), 2., false),
        row(Some(1.5), 3., true),
        row(None, 3., false),
    ];
    let covariates = [-1., 0., 0., 1., 1., -1., 2., 0., 0., 2.];
    let design = MatrixView::from_slice(5, 2, &covariates)?;
    let cox = PreparedCox::new(
        &data,
        design,
        CoxTies::Efron,
        SurvivalOptions::new().with_max_bytes(1024),
    )?;
    let beta = [0.2, -0.3];
    let mut score = [0.; 2];
    let mut information = [0.; 4];
    FitDerivatives::gradient(&cox, &beta, &mut score)?;
    FitDerivatives::hessian(&cox, &beta, &mut information)?;
    let step = 1e-5;
    for col in 0..2 {
        let mut plus = beta;
        let mut minus = beta;
        plus[col] += step;
        minus[col] -= step;
        let difference = (cox.value(&plus, &mut ())? - cox.value(&minus, &mut ())?) / (2. * step);
        assert!((difference - score[col]).abs() < 1e-9);
        let mut score_plus = [0.; 2];
        let mut score_minus = [0.; 2];
        FitDerivatives::gradient(&cox, &plus, &mut score_plus)?;
        FitDerivatives::gradient(&cox, &minus, &mut score_minus)?;
        for row in 0..2 {
            let difference = (score_plus[row] - score_minus[row]) / (2. * step);
            assert!((difference - information[row * 2 + col]).abs() < 1e-8);
        }
    }
    Ok(())
}

#[test]
fn single_precision_survival_and_cox() -> Result<(), SciOxideError> {
    let data = [
        SurvivalObservation::new(None, 1.0_f32, scioxide_stats::SurvivalOutcome::Event)?,
        SurvivalObservation::new(None, 2.0_f32, scioxide_stats::SurvivalOutcome::Censored)?,
        SurvivalObservation::new(
            Some(0.5_f32),
            2.0_f32,
            scioxide_stats::SurvivalOutcome::Event,
        )?,
    ];
    let km = kaplan_meier_with_options(
        &data,
        ConfidenceProbability::default(),
        scioxide_stats::SurvivalOptions::new().with_max_bytes(1024),
    )?;
    assert!((km[0].survival - 2.0 / 3.0).abs() < 1e-6);
    let na = nelson_aalen_with_options(
        &data,
        ConfidenceProbability::default(),
        scioxide_stats::SurvivalOptions::new().with_max_bytes(1024),
    )?;
    assert!((na[0].cumulative_hazard - 1.0 / 3.0).abs() < 1e-6);
    let covariates = [-1.0_f32, 0.0, 1.0];
    let design = MatrixView::from_slice(3, 1, &covariates)?;
    let cox = PreparedCox::new(
        &data,
        design,
        CoxTies::Breslow,
        SurvivalOptions::new().with_max_bytes(1024),
    )?;
    let value = cox.value(&[0.0], &mut ())?;
    assert!((value - 3.0_f32.ln() - 2.0_f32.ln()).abs() < 1e-5);
    Ok(())
}

#[test]
fn pointwise_kaplan_meier_coverage_under_independent_censoring() -> Result<(), SciOxideError> {
    let mut state = 0x5555_aaaa_1234_9876_u64;
    let mut covered = 0usize;
    let truth = (-1.0_f64).exp();
    for _ in 0..400 {
        let mut observations = Vec::new();
        for _ in 0..64 {
            let mut uniform = || {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((state >> 11) as f64 + 0.5) / ((1_u64 << 53) as f64)
            };
            let failure = -uniform().ln();
            let censor = -uniform().ln() / 0.6;
            observations.push(row(None, failure.min(censor), failure <= censor));
        }
        let curve = kaplan_meier_with_options(
            &observations,
            ConfidenceProbability::default(),
            scioxide_stats::SurvivalOptions::new().with_max_bytes(8192),
        )?;
        if let Some(interval) = curve
            .iter()
            .rev()
            .find(|point| point.time <= 1.)
            .and_then(|point| point.interval)
        {
            covered += usize::from(interval.contains(truth));
        }
    }
    assert!((350..=396).contains(&covered), "covered {covered}/400");
    Ok(())
}

fn curve_workspace<S: scioxide_core::RealScalar>(
    observations: &[scioxide_stats::SurvivalObservation<S>],
    kind: scioxide_stats::SurvivalEstimator,
) -> scioxide_stats::SurvivalWorkspace<S> {
    let req = scioxide_stats::SurvivalWorkspace::workspace_requirements(
        observations,
        kind,
        scioxide_stats::SurvivalOptions::new(),
    )
    .unwrap();
    scioxide_stats::SurvivalWorkspace::workspace(&req).unwrap()
}

#[test]
fn undersized_survival_workspaces_report_their_shortfall() -> Result<(), SciOxideError> {
    use scioxide_core::{Capacity, SciOxideErrorView};
    use scioxide_stats::{SurvivalEstimator, SurvivalWorkspace};
    let data = [
        row(None, 1., true),
        row(None, 2., false),
        row(Some(0.5), 3., true),
    ];
    let options = SurvivalOptions::new();
    let confidence = ConfidenceProbability::default();
    let context = ExecutionContext::sequential();
    let km =
        SurvivalWorkspace::workspace_requirements(&data, SurvivalEstimator::KaplanMeier, options)?;
    let na =
        SurvivalWorkspace::workspace_requirements(&data, SurvivalEstimator::NelsonAalen, options)?;
    let mut workspace = SurvivalWorkspace::workspace(&km)?;
    assert_eq!(workspace.check(&km)?, Capacity::Sufficient);
    assert_eq!(workspace.check(&na)?, Capacity::NeedsGrowth);
    let expected = nelson_aalen_with_options(&data, confidence, options)?;
    let mut output = expected.clone();
    output[0].cumulative_hazard = -1.;
    let error = nelson_aalen_into(
        &data,
        confidence,
        options,
        &mut output,
        &mut workspace,
        context,
    )
    .err()
    .unwrap();
    let SciOxideErrorView::ResourceLimit {
        resource,
        required,
        limit,
    } = error.view()
    else {
        panic!("expected a workspace shortfall, got {error}");
    };
    assert_eq!(resource.as_str(), "survival_workspace_points");
    assert_eq!((required, limit), (2, 0));
    assert_eq!(output[0].cumulative_hazard, -1.);
    workspace.try_reserve(&na)?;
    assert_eq!(workspace.check(&na)?, Capacity::Sufficient);
    assert_eq!(workspace.check(&km)?, Capacity::Sufficient);
    assert_eq!(
        nelson_aalen_into(
            &data,
            confidence,
            options,
            &mut output,
            &mut workspace,
            context
        )?,
        2
    );
    assert_eq!(output[1].cumulative_hazard, expected[1].cumulative_hazard);
    // Logrank and Cox preparation take their byte limits from the same options.
    let design = [0., 1., 2.];
    for result in [
        logrank_test_with_options(&data[..2], &data[1..], options.with_max_bytes(1)).err(),
        PreparedCox::new(
            &data,
            MatrixView::from_slice(3, 1, &design)?,
            CoxTies::Efron,
            options.with_max_bytes(1),
        )
        .err(),
    ] {
        let error = result.unwrap();
        let SciOxideErrorView::ResourceLimit {
            required, limit, ..
        } = error.view()
        else {
            panic!("expected a byte limit, got {error}");
        };
        assert!(required > limit && limit == 1);
    }
    Ok(())
}

#[test]
fn prepared_cox_clones_share_the_objective() -> Result<(), SciOxideError> {
    let data = [
        row(None, 1., true),
        row(None, 2., true),
        row(None, 3., false),
    ];
    let covariates = [0., 1., 3.];
    let design = MatrixView::from_slice(3, 1, &covariates)?;
    let model = PreparedCox::new(&data, design, CoxTies::Breslow, SurvivalOptions::new())?;
    let copy = model.clone();
    let (mut a, mut b) = ([0.], [0.]);
    model.gradient(&[0.25], &mut a)?;
    copy.gradient(&[0.25], &mut b)?;
    assert_eq!(a[0].to_bits(), b[0].to_bits());
    assert_eq!(copy.ties(), CoxTies::Breslow);
    Ok(())
}
