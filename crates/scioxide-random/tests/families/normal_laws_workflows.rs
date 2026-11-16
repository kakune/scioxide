//! normal laws workflows and statistical contracts.
#![allow(
    clippy::unwrap_used,
    clippy::cast_precision_loss,
    clippy::cast_lossless
)]
use scioxide_core::RealScalar;
use scioxide_random::*;

const NORMAL_STAT_SAMPLES: usize = 1_000_000;

const NORMAL_GOF_BINS: usize = 64;

fn assert_normal_ziggurat_statistics<S>()
where
    S: RealScalar,
{
    let dist = Normal::new(S::ZERO, S::ONE).unwrap();
    let mut rng = SplitMix64::from_seed(0x87);
    let mut samples = vec![S::ZERO; NORMAL_STAT_SAMPLES];
    let mut workspace = SamplingWorkspace::new();
    dist.sample_n_into(
        &mut rng,
        scioxide_random::SamplingPolicy::advance_on_error(scioxide_random::OutputPolicy::Atomic),
        &mut samples,
        &mut workspace,
    )
    .unwrap();

    let mut values: Vec<f64> = samples
        .into_iter()
        .map(|value| value.try_to_f64_exact().unwrap())
        .collect();
    assert!(values.iter().all(|value| value.is_finite()));

    let count = NORMAL_STAT_SAMPLES as f64;
    let mean = values.iter().sum::<f64>() / count;
    let variance = values
        .iter()
        .map(|value| {
            let centered = value - mean;
            centered * centered
        })
        .sum::<f64>()
        / count;
    let mean_bound = 8.0 / count.sqrt();
    let variance_bound = 8.0 * (2.0 / (count - 1.0)).sqrt();
    assert!(
        mean.abs() < mean_bound,
        "normal sample mean was {mean}, bound={mean_bound}"
    );
    assert!(
        (variance - 1.0).abs() < variance_bound,
        "normal sample variance was {variance}, bound={variance_bound}"
    );
    // Retain paired marginal and independence checks through consecutive
    // canonical samples, reusing the draws above.
    let pairs = (values.len() / 2) as f64;
    let (sx, sy, sxx, syy, sxy) = values.chunks_exact(2).fold(
        (0.0, 0.0, 0.0, 0.0, 0.0),
        |(sx, sy, sxx, syy, sxy), pair| {
            let [x, y] = [pair[0], pair[1]];
            (sx + x, sy + y, sxx + x * x, syy + y * y, sxy + x * y)
        },
    );
    let means = [sx / pairs, sy / pairs];
    let variances = [
        sxx / pairs - means[0].powi(2),
        syy / pairs - means[1].powi(2),
    ];
    for (axis, (&mean, &variance)) in means.iter().zip(&variances).enumerate() {
        assert!(
            mean.abs() < 8.0 / pairs.sqrt(),
            "normal pair {axis} mean was {mean}"
        );
        assert!(
            (variance - 1.0).abs() < 8.0 * (2.0 / (pairs - 1.0)).sqrt(),
            "normal pair {axis} variance was {variance}"
        );
    }
    let correlation = (sxy / pairs - means[0] * means[1]) / (variances[0] * variances[1]).sqrt();
    assert!(
        correlation.abs() < 8.0 / pairs.sqrt(),
        "consecutive normal samples have correlation {correlation}"
    );

    // Each two-sided tail is checked against an eight-binomial-sigma bound,
    // plus two observations of endpoint slack for finite sample counts.
    [
        (2.0, 0.045_500_263_896_358_42),
        (3.0, 0.002_699_796_063_260_191_3),
        (4.0, 0.000_063_342_483_666_239_84),
    ]
    .into_iter()
    .for_each(|(cutoff, expected)| {
        let observed = values.iter().filter(|value| value.abs() > cutoff).count() as f64 / count;
        let bound = 8.0 * (expected * (1.0 - expected) / count).sqrt() + 2.0 / count;
        assert!(
            (observed - expected).abs() < bound,
            "normal |x| > {cutoff} tail was {observed}, expected={expected}, bound={bound}"
        );
    });

    let boundaries = (1..NORMAL_GOF_BINS)
        .map(|index| {
            let probability = index as f64 / NORMAL_GOF_BINS as f64;
            dist.quantile(S::try_from_f64_lossy(probability).unwrap())
                .unwrap()
                .try_to_f64_exact()
                .unwrap()
        })
        .collect::<Vec<_>>();
    let mut samples = values.iter().copied();
    let bin_counts = statistics::histogram(&boundaries, values.len(), || samples.next().unwrap());
    let expected_bin_count = count / NORMAL_GOF_BINS as f64;
    let pearson = statistics::pearson(
        &bin_counts,
        core::iter::repeat_n(expected_bin_count, NORMAL_GOF_BINS),
    );
    // Chi-square(63) has mean 63 and standard deviation sqrt(126). This
    // threshold is more than eight standard deviations above the mean.
    assert!(
        pearson < 155.0,
        "normal 64-bin Pearson statistic was {pearson}"
    );

    values.sort_by(f64::total_cmp);
    [
        (0.001, -3.090_232_306_167_813, 0.08),
        (0.01, -2.326_347_874_040_840_8, 0.04),
        (0.5, 0.0, 0.015),
        (0.99, 2.326_347_874_040_840_8, 0.04),
        (0.999, 3.090_232_306_167_813, 0.08),
    ]
    .into_iter()
    .for_each(|(probability, expected, bound)| {
        let index = (probability * (count - 1.0)).round() as usize;
        let observed = values[index];
        assert!(
            (observed - expected).abs() < bound,
            "normal p={probability} quantile was {observed}, expected={expected}, bound={bound}"
        );
    });
}

#[test]
fn test_normal_ziggurat_statistics_f32() {
    assert_normal_ziggurat_statistics::<f32>();
}

#[test]
fn test_normal_ziggurat_statistics_f64() {
    assert_normal_ziggurat_statistics::<f64>();
}

#[test]
fn test_normal_cdf_at_mean() {
    let dist = Normal::new(3.0_f64, 2.0).unwrap();
    let cdf_at_mean = dist.cdf(3.0).unwrap();
    assert!(
        (cdf_at_mean - 0.5).abs() < 1e-10,
        "Normal CDF at mean should be 0.5, got {cdf_at_mean}"
    );
}

#[path = "../support/statistics.rs"]
mod statistics;
