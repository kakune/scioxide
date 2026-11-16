//! Right-censored estimators and an injected stationary-point Cox solver.
use scioxide_core::{EvaluationRole, RankThreshold, SciOxideError, SolveContext};
use scioxide_linalg::{HermitianEigenOptions, MatrixView};
use scioxide_stats::{
    ConfidenceProbability, CoxInferenceOptions, CoxTies, FitRequest, FitSolver,
    FitSolverCapabilities, PreparedCox, StatsTermination, SurvivalObservation,
    cox_fit_with_options, kaplan_meier_with_options, logrank_test_with_options,
    nelson_aalen_with_options,
};

// This example solver only accepts an already stationary one-parameter start.
// Production callers inject a solver with their documented stopping policy.
struct StationaryStart;

impl FitSolver<f64> for StationaryStart {
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
        if !context.record_iteration()?.is_granted()
            || !context
                .record_provider(EvaluationRole::Gradient)?
                .is_granted()
        {
            return Ok(StatsTermination::BudgetExhausted);
        }
        let mut gradient = [0.0];
        let derivatives = request
            .derivatives
            .ok_or_else(|| SciOxideError::external("example needs a score"))?;
        derivatives.gradient(candidate, &mut gradient)?;
        Ok(if gradient[0].abs() < 1e-12 {
            StatsTermination::Converged
        } else {
            StatsTermination::NotConverged
        })
    }
}

fn main() -> Result<(), SciOxideError> {
    let observations = [
        SurvivalObservation::new(None, 2.0, scioxide_stats::SurvivalOutcome::Censored)?,
        SurvivalObservation::new(None, 1.0, scioxide_stats::SurvivalOutcome::Event)?,
        SurvivalObservation::new(None, 2.0, scioxide_stats::SurvivalOutcome::Censored)?,
    ];
    let confidence = ConfidenceProbability::default();
    println!(
        "Kaplan–Meier: {:?}",
        kaplan_meier_with_options(
            &observations,
            confidence,
            scioxide_stats::SurvivalOptions::new().with_max_bytes(4096)
        )?
    );
    println!(
        "Nelson–Aalen: {:?}",
        nelson_aalen_with_options(
            &observations,
            confidence,
            scioxide_stats::SurvivalOptions::new().with_max_bytes(4096)
        )?
    );
    println!(
        "Logrank: {:?}",
        logrank_test_with_options(
            &observations[..2],
            &observations[2..],
            scioxide_stats::SurvivalOptions::new().with_max_bytes(4096)
        )?
    );

    let covariates = [-1.0, 0.0, 1.0];
    let design = MatrixView::from_slice(3, 1, &covariates)?;
    let problem = PreparedCox::new(
        &observations,
        design,
        CoxTies::Efron,
        scioxide_stats::SurvivalOptions::new().with_max_bytes(1024),
    )?;

    let fitted = cox_fit_with_options(
        &problem,
        &[0.0],
        &mut StationaryStart,
        CoxInferenceOptions::new()
            .with_confidence(confidence)
            .with_rank_tolerance(RankThreshold::new(0.0, 1e-10)?)
            .with_eigen_options(HermitianEigenOptions::default())
            .with_max_fit_bytes(1024)
            .with_max_report_bytes(4096),
    )?;
    println!("Cox coefficients: {:?}", fitted.fit().parameters());
    if let Some(report) = fitted.report() {
        println!("Cox baseline: {:?}", report.baseline());
        println!(
            "Schoenfeld residuals: {:?}",
            report
                .schoenfeld_residuals()
                .flat_map(|row| row.values().iter())
                .copied()
                .collect::<Vec<_>>()
        );
    }
    Ok(())
}
