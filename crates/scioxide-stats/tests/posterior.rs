#![allow(clippy::unwrap_used, clippy::panic)]

#[cfg(feature = "std")]
use scioxide_core::ThreadExecutor;
use scioxide_core::{ExecutionContext, NumericalIssue, SciOxideErrorView};
#[cfg(not(feature = "std"))]
#[path = "support/executor.rs"]
mod executor;
#[cfg(not(feature = "std"))]
use executor::PartitionExecutor;
use scioxide_linalg::{Matrix, MatrixView};
use scioxide_random::{Distribution, Normal, Pcg64, SeedableRng};
use scioxide_stats::{PosteriorOptions, posterior_summaries_into, posterior_summary_with_options};
use std::num::NonZeroUsize;

fn close(actual: f64, expected: f64, case: &str) {
    assert!(actual.is_finite(), "{case}: nonfinite {actual}");
    assert!(
        (actual - expected).abs() <= 3e-12 * expected.abs() + 4.0 * f64::from_bits(1),
        "{case}: actual {actual:.17e}, expected {expected:.17e}"
    );
}

#[test]
fn pinned_independent_diagnostic_and_exact_moment_fixtures() {
    for line in include_str!("data/posterior.tsv")
        .lines()
        .filter(|line| !line.starts_with('#'))
    {
        let fields: Vec<_> = line.split('\t').collect();
        let case = fields[0];
        let chains = fields[2].parse().unwrap();
        let draws = fields[3].parse().unwrap();
        let max_lag = fields[4].parse().unwrap();
        let options = PosteriorOptions::default().with_max_lag(max_lag);
        let samples: Vec<f64> = fields[5]
            .split(',')
            .map(|bits| {
                if fields[1] == "f32" {
                    f64::from(f32::from_bits(u32::from_str_radix(bits, 16).unwrap()))
                } else {
                    f64::from_bits(u64::from_str_radix(bits, 16).unwrap())
                }
            })
            .collect();
        let result = if fields[1] == "f32" {
            let samples: Vec<_> = samples.iter().map(|&x| x as f32).collect();
            posterior_summary_with_options(
                MatrixView::from_slice(chains, draws, &samples).unwrap(),
                options,
            )
        } else {
            posterior_summary_with_options(
                MatrixView::from_slice(chains, draws, &samples).unwrap(),
                options,
            )
        };
        if fields[7] == "inf" {
            assert!(
                matches!(
                    result.unwrap_err().view(),
                    SciOxideErrorView::NumericalInstability {
                        issue: NumericalIssue::Overflow,
                        ..
                    }
                ),
                "{case}"
            );
            continue;
        }
        let result = result.unwrap();
        for (actual, expected) in [
            result.mean,
            result.variance,
            result.quantiles[0],
            result.quantiles[1],
            result.quantiles[2],
        ]
        .into_iter()
        .zip(&fields[6..11])
        {
            close(actual, expected.parse().unwrap(), case);
        }
        for (actual, expected) in [
            result.rhat,
            result.bulk_ess,
            result.tail_ess,
            result.mean_mcse,
        ]
        .into_iter()
        .zip(&fields[11..15])
        {
            if *expected == "none" {
                assert_eq!(actual, None, "{case}");
            } else {
                close(
                    actual.unwrap_or_else(|| panic!("{case}: absent diagnostic")),
                    expected.parse().unwrap(),
                    case,
                );
            }
        }
        assert_eq!((result.chains, result.draws_per_chain), (chains, draws));
    }
}

#[test]
fn reported_regressions_and_underflowing_variance_retain_usable_diagnostics() {
    let alternating = [-1., 1., -1., 1., -1., 1., -1., 1.];
    let result = posterior_summary_with_options(
        MatrixView::from_slice(2, 4, &alternating).unwrap(),
        PosteriorOptions::default(),
    )
    .unwrap();
    close(result.rhat.unwrap(), 0.5_f64.sqrt(), "folded constant");
    close(
        result.bulk_ess.unwrap(),
        7.224719895935548,
        "short antithetic ESS",
    );
    let longer = alternating.repeat(4);
    let result = posterior_summary_with_options(
        MatrixView::from_slice(2, 16, &longer).unwrap(),
        PosteriorOptions::default(),
    )
    .unwrap();
    assert!(result.bulk_ess.unwrap() > longer.len() as f64);
    let large = [-2e154, 0., 0., 0., -2e154, 0., 0., 0.];
    let result = posterior_summary_with_options(
        MatrixView::from_slice(2, 4, &large).unwrap(),
        PosteriorOptions::default(),
    )
    .unwrap();
    close(result.mean, -5e153, "reported mean");
    close(result.variance, 8.571428571428571e307, "reported variance");
    assert!(result.mean_mcse.unwrap() > 0.);
    let tiny = large.map(|x| x / 2e154 * 2.0_f64.powi(-600));
    let result = posterior_summary_with_options(
        MatrixView::from_slice(2, 4, &tiny).unwrap(),
        PosteriorOptions::default(),
    )
    .unwrap();
    assert_eq!(result.variance, 0.);
    assert!(result.mean_mcse.unwrap() > 0.);
    assert!(result.bulk_ess.is_some());
}

#[test]
fn numerical_failure_is_transactional_in_sequential_and_partitioned_batches() {
    let good = [-2e154, 0., 0., 0., -2e154, 0., 0., 0.];
    let bad = [
        -f64::MAX,
        f64::MAX,
        -f64::MAX,
        f64::MAX,
        -f64::MAX,
        f64::MAX,
        -f64::MAX,
        f64::MAX,
    ];
    let good = MatrixView::from_slice(2, 4, &good).unwrap();
    let bad = MatrixView::from_slice(2, 4, &bad).unwrap();
    let expected = posterior_summary_with_options(good, PosteriorOptions::default()).unwrap();
    #[cfg(feature = "std")]
    let pool = ThreadExecutor::new(NonZeroUsize::new(2).unwrap());
    #[cfg(not(feature = "std"))]
    let pool = PartitionExecutor(NonZeroUsize::new(2).unwrap());
    for context in [ExecutionContext::sequential(), ExecutionContext::new(&pool)] {
        let mut output = [None, None];
        posterior_summaries_into(
            &[good, good],
            PosteriorOptions::default(),
            &mut output,
            context,
        )
        .unwrap();
        assert_eq!(output, [Some(expected); 2]);
        let before = output;
        let error = posterior_summaries_into(
            &[good, bad],
            PosteriorOptions::default(),
            &mut output,
            context,
        )
        .unwrap_err();
        assert!(matches!(
            error.view(),
            SciOxideErrorView::NumericalInstability {
                issue: NumericalIssue::Overflow,
                ..
            }
        ));
        assert_eq!(output, before);
    }
}

#[test]
fn independent_and_correlated_chains_have_distinct_effective_sizes() {
    let mut rng = Pcg64::from_seed(558);
    let normal = Normal::<f64>::new(0.0, 1.0).unwrap();
    let mut independent = Vec::new();
    let mut correlated = Vec::new();
    for _ in 0..4 {
        let mut state = 0.0;
        for _ in 0..256 {
            let draw = normal.sample(&mut rng).unwrap();
            independent.push(draw);
            state = 0.9 * state + draw;
            correlated.push(state);
        }
    }
    let options = PosteriorOptions::default();
    let iid = posterior_summary_with_options(
        MatrixView::from_slice(4, 256, &independent).unwrap(),
        options,
    )
    .unwrap();
    let ar = posterior_summary_with_options(
        MatrixView::from_slice(4, 256, &correlated).unwrap(),
        options,
    )
    .unwrap();
    assert!(iid.rhat.unwrap() < 1.1);
    assert!(iid.bulk_ess.unwrap() > ar.bulk_ess.unwrap());
    assert!(ar.mean_mcse.unwrap() > iid.mean_mcse.unwrap());
    assert!(iid.quantiles[0] < iid.quantiles[1] && iid.quantiles[1] < iid.quantiles[2]);
}

#[test]
fn split_rhat_detects_chain_shift_and_constant_diagnostics_are_absent() {
    let mut draws = Vec::new();
    for chain in 0..4 {
        for i in 0..32 {
            draws.push((i % 8) as f64 + if chain == 3 { 20.0 } else { 0.0 });
        }
    }
    let report = posterior_summary_with_options(
        MatrixView::from_slice(4, 32, &draws).unwrap(),
        PosteriorOptions::default(),
    )
    .unwrap();
    assert!(report.rhat.unwrap() > 1.2);
    let constants = vec![2.0_f64; 32];
    let constant = posterior_summary_with_options(
        MatrixView::from_slice(2, 16, &constants).unwrap(),
        PosteriorOptions::default(),
    )
    .unwrap();
    assert_eq!(constant.mean, 2.0);
    assert_eq!(constant.variance, 0.0);
    assert_eq!(constant.rhat, None);
    assert_eq!(constant.bulk_ess, None);
}

#[test]
fn f32_batch_and_limits() {
    let samples: Vec<f32> = (0..64).map(|i| (i % 16) as f32).collect();
    let view = MatrixView::from_slice(4, 16, &samples).unwrap();
    let standalone = posterior_summary_with_options(view, PosteriorOptions::default()).unwrap();
    let owned = Matrix::from_vec(4, 16, samples.clone()).unwrap();
    assert_eq!(
        posterior_summary_with_options(owned.view(), PosteriorOptions::default()).unwrap(),
        standalone
    );
    let mut padded = vec![-99.0_f32; 128];
    for chain in 0..4 {
        for draw in 0..16 {
            padded[chain * 32 + draw * 2] = samples[chain * 16 + draw];
        }
    }
    let strided = MatrixView::from_slice_with_strides(4, 16, &padded, 32, 2).unwrap();
    assert_eq!(
        posterior_summary_with_options(strided, PosteriorOptions::default()).unwrap(),
        standalone
    );
    let mut output = [None, None];
    posterior_summaries_into(
        &[view, view],
        PosteriorOptions::default(),
        &mut output,
        ExecutionContext::sequential(),
    )
    .unwrap();
    assert_eq!(output, [Some(standalone), Some(standalone)]);
    #[cfg(feature = "std")]
    let pool = ThreadExecutor::new(NonZeroUsize::new(2).unwrap());
    #[cfg(not(feature = "std"))]
    let pool = PartitionExecutor(NonZeroUsize::new(2).unwrap());
    posterior_summaries_into(
        &[view, view],
        PosteriorOptions::default(),
        &mut output,
        ExecutionContext::new(&pool),
    )
    .unwrap();
    assert_eq!(output, [Some(standalone), Some(standalone)]);
    assert!(
        posterior_summary_with_options(
            view,
            PosteriorOptions::default().with_max_workspace_bytes(8)
        )
        .is_err()
    );
    assert!(
        posterior_summary_with_options(view, PosteriorOptions::default().with_max_lag_work(8))
            .is_err()
    );
    let mut short_output = [None];
    assert!(
        posterior_summaries_into(
            &[view, view],
            PosteriorOptions::default(),
            &mut short_output,
            ExecutionContext::sequential(),
        )
        .is_err()
    );
    let bad = [f32::NAN; 64];
    let invalid_view = MatrixView::from_slice(4, 16, &bad).unwrap();
    let mut sentinel = [Some(standalone), Some(standalone)];
    assert!(
        posterior_summaries_into(
            &[view, invalid_view],
            PosteriorOptions::default(),
            &mut sentinel,
            ExecutionContext::sequential(),
        )
        .is_err()
    );
    assert_eq!(sentinel, [Some(standalone), Some(standalone)]);
}
