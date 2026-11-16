//! Public generic empirical-risk workflow without downstream dependencies.

#![allow(clippy::unwrap_used)]

use scioxide_core::{RealScalar, SciOxideError};
use scioxide_stats::*;

#[derive(Debug, Clone, Copy)]
struct EmpiricalRiskSummary<S: RealScalar> {
    threshold: QuantileResult<S>,
    upper_tail: TailMeanResult<S>,
}

fn summarize_empirical_risk<S: RealScalar>(
    outcomes: &[S],
    probability: QuantileProbability,
) -> Result<EmpiricalRiskSummary<S>, SciOxideError> {
    let distribution = EmpiricalDistribution::new(outcomes, Unweighted)?;
    Ok(EmpiricalRiskSummary {
        threshold: distribution
            .quantile(probability, QuantileRule::new(QuantileMethod::NearestRank))?,
        upper_tail: distribution.tail_mean(probability, TailSide::Upper)?,
    })
}

fn summarize_weighted_empirical_risk<S: RealScalar>(
    weighted_outcomes: &[(S, S)],
    probability: QuantileProbability,
) -> Result<EmpiricalRiskSummary<S>, SciOxideError> {
    let distribution = {
        let pairs = weighted_outcomes;
        let values: Vec<_> = pairs.iter().map(|&(x, _)| x).collect();
        let masses: Vec<_> = pairs.iter().map(|&(_, w)| w).collect();
        EmpiricalDistribution::new(&values, Weights::<_, Mass>::new(&masses))
    }?;
    Ok(EmpiricalRiskSummary {
        threshold: distribution.quantile(probability, QuantileRule::inverse_ecdf())?,
        upper_tail: distribution.tail_mean(probability, TailSide::Upper)?,
    })
}

#[test]
fn generic_empirical_risk_workflow_records_conventions() {
    let probability = QuantileProbability::new(0.75).unwrap();
    let f64_summary = summarize_empirical_risk(&[1.0_f64, 2.0, 4.0, 8.0], probability).unwrap();
    assert_eq!(f64_summary.threshold.value(), 4.0);
    assert_eq!(f64_summary.upper_tail.mean(), 8.0);
    assert_eq!(
        f64_summary.threshold.convention(),
        QuantileConvention::Unweighted(QuantileMethod::NearestRank)
    );
    assert_eq!(
        f64_summary.upper_tail.convention(),
        TailMeanConvention::FractionalInverseEcdf
    );

    let f32_summary = summarize_empirical_risk(&[1.0_f32, 2.0, 4.0, 8.0], probability).unwrap();
    assert_eq!(f32_summary.threshold.value(), 4.0);
    assert_eq!(f32_summary.upper_tail.mean(), 8.0);

    let weighted =
        summarize_weighted_empirical_risk(&[(1.0_f64, 1.0), (4.0, 2.0), (8.0, 1.0)], probability)
            .unwrap();
    assert_eq!(weighted.threshold.value(), 4.0);
    assert_eq!(weighted.upper_tail.mean(), 8.0);
    assert_eq!(
        weighted.threshold.convention(),
        QuantileConvention::MassInverseEcdf
    );
}
