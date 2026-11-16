//! Generic empirical threshold and upper-tail summary.

use scioxide_core::{RealScalar, SciOxideError};
use scioxide_stats::{
    EmpiricalDistribution, QuantileMethod, QuantileProbability, QuantileResult, QuantileRule,
    TailMeanResult, TailSide, Unweighted,
};

fn summarize_empirical_risk<S: RealScalar>(
    outcomes: &[S],
    probability: QuantileProbability,
) -> Result<(QuantileResult<S>, TailMeanResult<S>), SciOxideError> {
    let distribution = EmpiricalDistribution::new(outcomes, Unweighted)?;
    let threshold =
        distribution.quantile(probability, QuantileRule::new(QuantileMethod::NearestRank))?;
    let upper_tail = distribution.tail_mean(probability, TailSide::Upper)?;
    Ok((threshold, upper_tail))
}

fn main() -> Result<(), SciOxideError> {
    let probability = QuantileProbability::new(0.95)?;
    let outcomes = [1.0_f64, 1.5, 2.0, 2.5, 4.0, 8.0];
    let (threshold, upper_tail) = summarize_empirical_risk(&outcomes, probability)?;

    println!(
        "threshold={:?} convention={:?}",
        threshold.value(),
        threshold.convention()
    );
    println!(
        "upper_tail_mean={:?} mass={:?} boundary_contribution={:?} convention={:?}",
        upper_tail.mean(),
        upper_tail.tail_mass(),
        upper_tail.boundary_contribution(),
        upper_tail.convention()
    );
    Ok(())
}
