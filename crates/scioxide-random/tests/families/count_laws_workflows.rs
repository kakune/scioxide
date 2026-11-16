//! count laws workflows and statistical contracts.
#![allow(
    clippy::unwrap_used,
    clippy::cast_precision_loss,
    clippy::cast_lossless
)]
use scioxide_core::RealScalar;
use scioxide_random::*;

const DISTRIBUTION_STAT_SAMPLES: usize = 100_000;

fn assert_probability_observation(label: &str, observed_count: usize, expected_probability: f64) {
    let sample_count = DISTRIBUTION_STAT_SAMPLES as f64;
    let observed_probability = observed_count as f64 / sample_count;
    let standard_error =
        (expected_probability * (1.0 - expected_probability) / sample_count).sqrt();
    let tolerance = 8.0 * standard_error + 2.0 / sample_count;

    assert!(
        (observed_probability - expected_probability).abs() <= tolerance,
        "{label}: observed probability {observed_probability}, expected {expected_probability}, tolerance {tolerance}"
    );
}

fn assert_poisson_statistics<S>(lambda: f64, seed: u64, verify_mass_regions: bool)
where
    S: RealScalar,
{
    let lambda_scalar = S::try_from_f64_lossy(lambda).unwrap();
    let dist = Poisson::new(lambda_scalar).unwrap();
    let sampler = dist.prepare_sampler().unwrap();
    let mut rng = SplitMix64::from_seed(seed);
    let mode = lambda.floor() as usize;
    let lower_tail_cutoff = (lambda - 2.0 * lambda.sqrt()).floor().max(0.0) as usize;
    let upper_tail_cutoff = (lambda + 2.0 * lambda.sqrt()).ceil() as usize;

    let (sum, sum_squares, mode_count, cdf_count, lower_tail_count, upper_tail_count) =
        (0..DISTRIBUTION_STAT_SAMPLES).fold(
            (0.0, 0.0, 0, 0, 0, 0),
            |(sum, sum_squares, mode_count, cdf_count, lower_tail_count, upper_tail_count), _| {
                let sample = sampler.sample(&mut rng).unwrap();
                let sample_f64 = sample as f64;
                (
                    sum + sample_f64,
                    sum_squares + sample_f64 * sample_f64,
                    mode_count + usize::from(sample == mode),
                    cdf_count + usize::from(sample <= mode),
                    lower_tail_count + usize::from(sample <= lower_tail_cutoff),
                    upper_tail_count + usize::from(sample >= upper_tail_cutoff),
                )
            },
        );

    let sample_count = DISTRIBUTION_STAT_SAMPLES as f64;
    let mean = sum / sample_count;
    let variance = sum_squares / sample_count - mean * mean;
    let mean_tolerance = 8.0 * (lambda / sample_count).sqrt();
    let variance_tolerance = 8.0 * ((lambda + 2.0 * lambda * lambda) / sample_count).sqrt();

    assert!(
        (mean - lambda).abs() <= mean_tolerance,
        "Poisson({lambda}) mean {mean}, tolerance {mean_tolerance}"
    );
    assert!(
        (variance - lambda).abs() <= variance_tolerance,
        "Poisson({lambda}) variance {variance}, tolerance {variance_tolerance}"
    );

    if verify_mass_regions {
        let expected_mode_mass = dist.pmf(mode).unwrap().try_to_f64_exact().unwrap();
        let expected_cdf = dist.cdf(mode).unwrap().try_to_f64_exact().unwrap();
        let expected_lower_tail = dist
            .cdf(lower_tail_cutoff)
            .unwrap()
            .try_to_f64_exact()
            .unwrap();
        let expected_upper_tail = 1.0
            - dist
                .cdf(upper_tail_cutoff - 1)
                .unwrap()
                .try_to_f64_exact()
                .unwrap();

        assert_probability_observation("mode PMF", mode_count, expected_mode_mass);
        assert_probability_observation("CDF at the mode", cdf_count, expected_cdf);
        assert_probability_observation(
            "lower two-sigma tail",
            lower_tail_count,
            expected_lower_tail,
        );
        assert_probability_observation(
            "upper two-sigma tail",
            upper_tail_count,
            expected_upper_tail,
        );
    } else {
        const NORMAL_TWO_SIGMA_TAIL: f64 = 0.022_750_131_948_179_21;
        let lower_tail = lower_tail_count as f64 / sample_count;
        let upper_tail = upper_tail_count as f64 / sample_count;
        assert!(
            (lower_tail - NORMAL_TWO_SIGMA_TAIL).abs() < 0.006,
            "Poisson({lambda}) lower-tail probability was {lower_tail}"
        );
        assert!(
            (upper_tail - NORMAL_TWO_SIGMA_TAIL).abs() < 0.006,
            "Poisson({lambda}) upper-tail probability was {upper_tail}"
        );
    }
}

fn assert_poisson_sampling_regimes<S>()
where
    S: RealScalar,
{
    for (lambda, seed, verify_mass_regions) in [
        (3.0, 0x310, true),
        (9.999, 0x999, true),
        (10.0, 0x1_000, true),
        (11.999, 0x11_999, true),
        (12.0, 0x12_000, true),
        (1_000.0, 0x1_000, true),
        (1_000_000.0, 0x10_000, false),
    ] {
        assert_poisson_statistics::<S>(lambda, seed, verify_mass_regions);
    }
}

#[test]
fn test_poisson_sampling_regimes_f32() {
    assert_poisson_sampling_regimes::<f32>();
}

#[test]
fn test_poisson_sampling_regimes_f64() {
    assert_poisson_sampling_regimes::<f64>();
}

#[test]
fn test_poisson_large_lambda_uniform_work_is_bounded() {
    const SAMPLE_COUNT: usize = 10_000;

    for lambda in [12.0_f64, 100.0, 10_000.0, 1_000_000.0, 1_000_000_000.0] {
        let dist = Poisson::new(lambda).unwrap();
        let sampler = dist.prepare_sampler().unwrap();
        let mut rng = SplitMix64::from_seed(0x88);
        let start = rng.position().unwrap();
        for _ in 0..SAMPLE_COUNT {
            let _ = sampler.sample(&mut rng).unwrap();
        }
        let draws_per_sample = (rng.position().unwrap() - start) as f64 / SAMPLE_COUNT as f64;
        assert!(
            draws_per_sample < 8.0,
            "Poisson({lambda}) consumed {draws_per_sample} uniforms per sample"
        );
    }
}

fn assert_binomial_statistics<S>(
    trials: usize,
    probability: f64,
    seed: u64,
    verify_mass_regions: bool,
) where
    S: RealScalar,
{
    let probability_scalar = S::try_from_f64_lossy(probability).unwrap();
    let probability = probability_scalar.try_to_f64_exact().unwrap();
    let dist = Binomial::new(trials, probability_scalar).unwrap();
    let sampler = dist.prepare_sampler().unwrap();
    let trials_f64 = trials as f64;
    let expected_mean = trials_f64 * probability;
    let expected_variance = expected_mean * (1.0 - probability);
    let standard_deviation = expected_variance.sqrt();
    let mode = ((trials_f64 + 1.0) * probability).floor() as usize;
    let lower_tail_cutoff = (expected_mean - 2.0 * standard_deviation).floor().max(0.0) as usize;
    let upper_tail_cutoff = (expected_mean + 2.0 * standard_deviation)
        .ceil()
        .min(trials_f64) as usize;
    let mut rng = SplitMix64::from_seed(seed);

    let (
        sum_deviations,
        sum_squared_deviations,
        mode_count,
        cdf_count,
        lower_tail_count,
        upper_tail_count,
    ) = (0..DISTRIBUTION_STAT_SAMPLES).fold(
        (0.0, 0.0, 0, 0, 0, 0),
        |(
            sum_deviations,
            sum_squared_deviations,
            mode_count,
            cdf_count,
            lower_tail_count,
            upper_tail_count,
        ),
         _| {
            let sample = sampler.sample(&mut rng).unwrap();
            assert!(sample <= trials);
            let deviation = sample as f64 - expected_mean;
            (
                sum_deviations + deviation,
                sum_squared_deviations + deviation * deviation,
                mode_count + usize::from(sample == mode),
                cdf_count + usize::from(sample <= mode),
                lower_tail_count + usize::from(sample <= lower_tail_cutoff),
                upper_tail_count + usize::from(sample >= upper_tail_cutoff),
            )
        },
    );

    let sample_count = DISTRIBUTION_STAT_SAMPLES as f64;
    let mean_deviation = sum_deviations / sample_count;
    let mean = expected_mean + mean_deviation;
    let variance = sum_squared_deviations / sample_count - mean_deviation * mean_deviation;
    let mean_tolerance = 8.0 * (expected_variance / sample_count).sqrt() + 2.0 / sample_count;
    let fourth_central_moment = 3.0 * expected_variance * expected_variance
        + expected_variance * (1.0 - 6.0 * probability * (1.0 - probability));
    let variance_tolerance = 8.0
        * ((fourth_central_moment - expected_variance * expected_variance) / sample_count)
            .max(0.0)
            .sqrt()
        + 2.0 / sample_count;

    assert!(
        (mean - expected_mean).abs() <= mean_tolerance,
        "Binomial({trials}, {probability}) mean {mean}, expected {expected_mean}, tolerance {mean_tolerance}"
    );
    assert!(
        (variance - expected_variance).abs() <= variance_tolerance,
        "Binomial({trials}, {probability}) variance {variance}, expected {expected_variance}, tolerance {variance_tolerance}"
    );

    if verify_mass_regions {
        let expected_mode_mass = dist.pmf(mode).unwrap().try_to_f64_exact().unwrap();
        let expected_cdf = dist.cdf(mode).unwrap().try_to_f64_exact().unwrap();
        let expected_lower_tail = dist
            .cdf(lower_tail_cutoff)
            .unwrap()
            .try_to_f64_exact()
            .unwrap();
        let expected_upper_tail = 1.0
            - dist
                .cdf(upper_tail_cutoff - 1)
                .unwrap()
                .try_to_f64_exact()
                .unwrap();

        assert_probability_observation("Binomial mode PMF", mode_count, expected_mode_mass);
        assert_probability_observation("Binomial CDF at the mode", cdf_count, expected_cdf);
        assert_probability_observation(
            "Binomial lower two-sigma tail",
            lower_tail_count,
            expected_lower_tail,
        );
        assert_probability_observation(
            "Binomial upper two-sigma tail",
            upper_tail_count,
            expected_upper_tail,
        );
    }
}

fn assert_binomial_sampling_regimes<S>()
where
    S: RealScalar,
{
    for (trials, probability, seed, verify_mass_regions) in [
        (8_usize, 0.25_f64, 0x8, true),
        (16, 0.5, 0x16, true),
        (17, 0.5, 0x17, true),
        (1_000, 0.009_999, 0x999, true),
        (1_000, 0.010_001, 0x1_001, true),
        (1_000, 0.5, 0x1_000, true),
        (1_000_000, 0.25, 0x10_000, false),
        (1_000_000, 0.000_001, 0x10_001, false),
        (1_000_000, 0.999_999, 0x10_002, false),
    ] {
        assert_binomial_statistics::<S>(trials, probability, seed, verify_mass_regions);
    }
}

#[test]
fn test_binomial_sampling_regimes_f32() {
    assert_binomial_sampling_regimes::<f32>();
}

#[test]
fn test_binomial_sampling_regimes_f64() {
    assert_binomial_sampling_regimes::<f64>();
}

#[test]
fn test_binomial_large_n_uniform_work_is_bounded() {
    const SAMPLE_COUNT: usize = 10_000;

    for (trials, probability) in [
        (100_usize, 0.25_f64),
        (10_000, 0.25),
        (1_000_000, 0.25),
        (1_000_000_000, 0.25),
        (1_000, 0.001),
        (1_000_000, 0.000_001),
        (1_000_000_000, 0.000_000_001),
    ] {
        let dist = Binomial::new(trials, probability).unwrap();
        let sampler = dist.prepare_sampler().unwrap();
        let mut rng = SplitMix64::from_seed(0x89);
        let start = rng.position().unwrap();
        for _ in 0..SAMPLE_COUNT {
            let _ = sampler.sample(&mut rng).unwrap();
        }
        let draws_per_sample = (rng.position().unwrap() - start) as f64 / SAMPLE_COUNT as f64;
        assert!(
            draws_per_sample < 8.0,
            "Binomial({trials}, {probability}) consumed {draws_per_sample} uniforms per sample"
        );
    }
}

#[test]
fn test_binomial_pmf_sums_to_one() {
    let dist = Binomial::new(15_usize, 0.3_f64).unwrap();
    let sum: f64 = (0..=15).map(|k| dist.pmf(k).unwrap()).sum();
    assert!(
        (sum - 1.0).abs() < 1e-10,
        "Binomial PMF should sum to 1, got {sum}"
    );
}

#[test]
fn test_poisson_pmf_sums_near_one() {
    let dist = Poisson::new(8.0_f64).unwrap();
    let sum: f64 = (0..40).map(|k| dist.pmf(k).unwrap()).sum();
    assert!(
        (sum - 1.0).abs() < 1e-8,
        "Poisson PMF should sum to ~1, got {sum}"
    );
}
