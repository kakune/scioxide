//! Descriptive moments, explicit weight models, covariance and trimmed reductions.

use scioxide_core::SciOxideError;
use scioxide_stats::*;

fn main() -> Result<(), SciOxideError> {
    let observations = [1.0_f64, 2.0, 4.0, 8.0, 16.0, 32.0];
    let merged = observations
        .chunks(2)
        .map(|partition| {
            partition.iter().copied().try_fold(
                OnlineMoments::new(MomentOrder::Fourth),
                |mut moments, value| {
                    moments.observe(value)?;
                    Ok::<_, SciOxideError>(moments)
                },
            )
        })
        .try_fold(
            OnlineMoments::new(MomentOrder::Fourth),
            |mut merged, partial| {
                merged.merge(&partial?)?;
                Ok::<_, SciOxideError>(merged)
            },
        )?;
    let snapshot = merged.snapshot();
    let confidence_interval = snapshot.confidence_interval_of_mean_with_options(
        ConfidenceProbability::new(0.95)?,
        scioxide_stats::MeanIntervalMethod::StudentT,
        scioxide_special::BetaOptions::default(),
    )?;
    if let Some(interval) = confidence_interval {
        println!(
            "method={:?} degrees={:?} assumptions={:?} bounds={:?}",
            interval.method(),
            interval.degrees_of_freedom(),
            interval.assumptions(),
            interval.bounds()
        );
    }
    println!(
        "count={} mean={:?} sample_variance={:?} interval={confidence_interval:?}",
        snapshot.count(),
        snapshot.mean()?,
        snapshot.sample_variance()?
    );
    println!(
        "skew={:?} excess={:?} k4={:?} sem={:?}",
        snapshot.skewness(BiasCorrection::Corrected)?,
        snapshot.kurtosis(BiasCorrection::Corrected, KurtosisConvention::Fisher)?,
        snapshot.k_statistic(4)?,
        snapshot.standard_error(1)?
    );

    fn report<M: MomentWeights>() -> Result<WeightedMomentsSnapshot<f64, M>, SciOxideError> {
        let weighted = [(1.0_f64, 1.0), (3.0, 2.0), (5.0, 1.0)]
            .into_iter()
            .try_fold(
                OnlineWeightedMoments::<f64, M>::new(MomentOrder::Fourth),
                |mut moments, (value, weight)| {
                    moments.observe(value, weight)?;
                    Ok::<_, SciOxideError>(moments)
                },
            )?
            .snapshot();
        println!(
            "{:?}: weighted_mean={:?} effective_sample_size={:?} degrees={:?} mean_variance={:?}",
            weighted.weight_semantics(),
            weighted.mean()?,
            weighted.effective_sample_size()?,
            weighted.degrees_of_freedom()?,
            weighted.variance_of_weighted_mean()?
        );
        Ok(weighted)
    }
    println!("frequency k2={:?}", report::<Frequency>()?.k_statistic(2)?);
    let _ = report::<Reliability>()?;
    println!(
        "analytic noise_scale={:?}",
        report::<Analytic>()?.noise_scale()?
    );

    let covariance = observations
        .into_iter()
        .zip(observations.into_iter().rev())
        .try_fold(
            OnlineCovariance::new(MomentOrder::Fourth),
            |mut covariance, (x, y)| {
                covariance.observe(x, y)?;
                Ok::<_, SciOxideError>(covariance)
            },
        )?
        .snapshot();
    println!(
        "sample_covariance_with_options={:?} correlation={:?} validity={:?}",
        covariance.sample_covariance()?,
        covariance.correlation()?,
        covariance.correlation_validity()
    );
    let trimmed = reduce_trimmed_moments_with_options(
        BatchView::new(
            &observations,
            scioxide_core::StridedLayout::new(&[1, observations.len()], &[observations.len(), 1])?,
            Axis::All,
        )?
        .with_axis(Axis::All)?,
        TrimLimits::new(Some(2.0), Some(16.0))?,
        StatisticsOptions::new().with_moment_order(MomentOrder::Fourth),
    )?;
    println!(
        "trimmed_mean={:?} trimmed_sem={:?}",
        trimmed[0].mean()?,
        trimmed[0].standard_error(1)?
    );
    Ok(())
}
