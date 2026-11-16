//! Location, rank and multiscale inference with explicit resampling resources.
use scioxide_core::*;
use scioxide_linalg::Matrix;
use scioxide_random::{OutputPolicy, SamplingPolicy, SeedableRng, SplitMix64};
use scioxide_stats::*;
fn main() -> Result<(), SciOxideError> {
    let x = [1., 1., 2., 4., 7.];
    let y = [0., 2., 2., 5., 3.];
    let mean = t_test_one_sample(&x, 0.)?;
    println!(
        "Mean contrast: {:?}; p={:?}; interval={:?}",
        mean.effect(),
        mean.p_value(),
        mean.interval()
    );
    let rank = mann_whitney_test(&x, &y)?;
    println!(
        "Mann–Whitney: statistic={}, p={:?}, method={:?}",
        rank.statistic(),
        rank.p_value(),
        rank.method()
    );
    let bws = BwsPlan::new(&x, &y, TestAlternative::TwoSided)?;
    let plan = bws.permutation_plan(
        PermutationMode::Randomized,
        ResamplingOptions::default()
            .with_replicates(999)
            .with_batch_size(4),
    )?;

    let report = bws.test(
        &plan,
        scioxide_stats::ReplicateOptions::sequential(
            &mut SplitMix64::from_seed(546),
            SamplingPolicy::advance_on_error(OutputPolicy::Atomic),
        ),
    )?;
    println!(
        "BWS: {:?}; completion={:?}",
        report.test(),
        report.resampling().termination()
    );
    let a = Matrix::from_vec(5, 1, vec![0., 1., 2., 3., 4.])?;
    let b = Matrix::from_vec(5, 1, vec![0., 1., 4., 9., 16.])?;
    let mgc = MgcPlan::from_samples(
        a.view(),
        b.view(),
        ClassicalOptions::default().with_max_workspace_bytes(1 << 20),
    )?;
    let mut scratch = mgc.workspace()?;
    println!(
        "MGC coefficient and selected scale: {:?}",
        mgc.statistic(None, &mut scratch)?
    );
    Ok(())
}
