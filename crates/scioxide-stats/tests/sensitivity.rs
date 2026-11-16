use scioxide_core::ExecutionContext;
#[cfg(feature = "std")]
use scioxide_core::ThreadExecutor;
use scioxide_random::{
    Pcg64, Philox4x64, QmcPlan, QmcReplicateOptions, QmcReplicatePlan, Rng, SeedableRng,
};
use scioxide_stats::{
    SobolEvaluations, sobol_bootstrap_scoped, sobol_indices, sobol_indices_monte_carlo_scoped,
    sobol_indices_qmc_scoped,
};
use std::num::NonZeroUsize;

#[allow(clippy::unwrap_used)]
fn balanced_scales<S: scioxide_core::RealScalar>(scales: &[S]) {
    for &s in scales {
        assert!(s > S::ZERO);
        let a = [-s, -s, s, s];
        let b = [-s, s, -s, s];
        let result = sobol_indices(SobolEvaluations {
            a: &a,
            b: &b,
            hybrids: &b,
            dimensions: 1,
        })
        .unwrap();
        assert!((result.first_order()[0] - 1.0).abs() < 2e-14);
        assert!((result.total_order()[0] - 1.0).abs() < 2e-14);
    }
}

#[test]
fn balanced_indices_survive_huge_and_subnormal_outputs() {
    balanced_scales(&[1.0_f64, 1e155, f64::MAX, 1e-200, f64::from_bits(1)]);
    balanced_scales(&[1.0_f32, 1e30, f32::MAX, 1e-30, f32::from_bits(1)]);
}

#[allow(clippy::unwrap_used)]
fn interaction_indices<S: scioxide_core::RealScalar>(scales: &[S], offset: S) {
    for &scale in scales {
        assert!(scale > S::ZERO);
        for shift in [S::ZERO, offset] {
            // Orthogonal Rademacher terms have variances 1, 4 and 9.
            let mut a = Vec::new();
            let mut b = Vec::new();
            let mut h = Vec::new();
            let model = |x: f64, y: f64| {
                shift + scale * S::try_from_f64_lossy(x + 2.0 * y + 3.0 * x * y).unwrap()
            };
            for ax in [-1.0, 1.0] {
                for ay in [-1.0, 1.0] {
                    for bx in [-1.0, 1.0] {
                        for by in [-1.0, 1.0] {
                            a.push(model(ax, ay));
                            b.push(model(bx, by));
                            h.extend([model(bx, ay), model(ax, by)]);
                        }
                    }
                }
            }
            // A translation is qualified only when it retains the sample contrasts.
            if a.iter().all(|&x| x == shift) {
                continue;
            }
            let fit = sobol_indices(SobolEvaluations {
                a: &a,
                b: &b,
                hybrids: &h,
                dimensions: 2,
            })
            .unwrap();
            let tolerance = if core::mem::size_of::<S>() == 4 {
                3e-6
            } else {
                3e-14
            };
            for (observed, expected) in fit.first_order().iter().chain(fit.total_order()).zip([
                1.0 / 14.0,
                4.0 / 14.0,
                10.0 / 14.0,
                13.0 / 14.0,
            ]) {
                assert!(
                    (observed - expected).abs() < tolerance,
                    "{observed} != {expected}"
                );
            }
        }
    }
}

#[test]
fn nonlinear_multi_input_indices_match_orthogonal_variance_components() {
    interaction_indices(
        &[1.0_f64, 2.0_f64.powi(510), f64::from_bits(16)],
        2.0_f64.powi(40),
    );
    interaction_indices(
        &[1.0_f32, 2.0_f32.powi(100), f32::from_bits(512)],
        2.0_f32.powi(16),
    );
}

#[test]
fn model_driven_iid_and_qmc_retain_output_units() {
    let root = Philox4x64::from_seed(1552);
    let plan = QmcReplicatePlan::new(
        &QmcPlan::sobol(4).unwrap(),
        &Philox4x64::from_seed(1552),
        QmcReplicateOptions::default()
            .with_replicates(4)
            .with_points_per_replicate(256),
    )
    .unwrap();
    let model = |x: &[f64]| {
        (if x[0] < 0.5 { -1.0 } else { 1.0 }) + 2.0 * (if x[1] < 0.5 { -1.0 } else { 1.0 })
    };
    let iid = |scale, shift| {
        sobol_indices_monte_carlo_scoped(
            2,
            256,
            &root,
            |_, rng: &mut Philox4x64| rng.next_float(),
            |x: &[f64]| Ok(shift + scale * model(x)),
            scioxide_stats::StatisticsOptions::new().with_max_bytes(32768),
            ExecutionContext::sequential(),
        )
        .unwrap()
    };
    let qmc = |scale, shift| {
        sobol_indices_qmc_scoped(
            &plan,
            2,
            256,
            |_, u: f64| Ok(u),
            |x| Ok(shift + scale * model(x)),
            scioxide_stats::StatisticsOptions::new().with_max_bytes(50000),
            ExecutionContext::sequential(),
        )
        .unwrap()
    };
    let base_iid = iid(1.0, 0.0);
    let base_qmc = qmc(1.0, 0.0);
    for (scale, shift) in [
        (2.0_f64.powi(510), 0.0),
        (f64::from_bits(16), 0.0),
        (1.0, 2.0_f64.powi(40)),
    ] {
        let observed_iid = iid(scale, shift);
        let observed_qmc = qmc(scale, shift);
        for (a, b) in [
            (base_iid.indices(), observed_iid.indices()),
            (&base_qmc, &observed_qmc),
        ] {
            for (a, b) in a
                .first_order()
                .iter()
                .chain(a.total_order())
                .zip(b.first_order().iter().chain(b.total_order()))
            {
                assert!((a - b).abs() < 3e-14);
            }
            match (a.randomized_standard_error(), b.randomized_standard_error()) {
                (Some((af, at)), Some((bf, bt))) => {
                    for (a, b) in af.iter().chain(at).zip(bf.iter().chain(bt)) {
                        assert!((a - b).abs() < 3e-14);
                    }
                }
                (None, None) => {}
                _ => assert_eq!(
                    a.randomized_standard_error().is_some(),
                    b.randomized_standard_error().is_some()
                ),
            }
        }
        for (&base, &observed) in base_iid
            .evaluations()
            .a
            .iter()
            .zip(observed_iid.evaluations().a)
        {
            assert_eq!(shift + scale * base, observed);
        }
    }
}

#[test]
fn centered_indices_and_paired_bootstrap_are_translation_invariant() {
    // Unbalanced finite design: pooled mean=1/2, variance=21/4.
    // Centered first numerator=39/8 and total numerator=23/8.
    let a = [-2.0, -1.0, 1.0, 2.0];
    let b = [3.0, -3.0, 4.0, 0.0];
    let h = [1.0, -2.0, 3.0, -1.0];
    let evaluations = SobolEvaluations {
        a: &a,
        b: &b,
        hybrids: &h,
        dimensions: 1,
    };
    let base = sobol_indices(evaluations).unwrap();
    assert!((base.first_order()[0] - 13.0 / 14.0).abs() < 1e-14);
    assert!((base.total_order()[0] - 23.0 / 42.0).abs() < 1e-14);
    for (scale, offset) in [
        (1.0, 2.0_f64.powi(40)),
        (-2.0, 16.0),
        (1e155, 0.0),
        (1e-200, 0.0),
    ] {
        let a = a.map(|x| offset + scale * x);
        let b = b.map(|x| offset + scale * x);
        let h = h.map(|x| offset + scale * x);
        let transformed = SobolEvaluations {
            a: &a,
            b: &b,
            hybrids: &h,
            dimensions: 1,
        };
        let fit = sobol_indices(transformed).unwrap();
        assert!((fit.first_order()[0] - base.first_order()[0]).abs() < 1e-13);
        assert!((fit.total_order()[0] - base.total_order()[0]).abs() < 1e-13);
        let root = Pcg64::from_seed(1549);
        let bootstrap = |e| {
            sobol_bootstrap_scoped(
                e,
                &root,
                16,
                scioxide_stats::ConfidenceProbability::new(0.9).unwrap(),
                scioxide_stats::StatisticsOptions::new().with_max_bytes(8192),
                ExecutionContext::sequential(),
            )
            .unwrap()
        };
        let expected = bootstrap(evaluations);
        let observed = bootstrap(transformed);
        for (left, right) in expected
            .first_order()
            .iter()
            .chain(expected.total_order())
            .zip(observed.first_order().iter().chain(observed.total_order()))
        {
            assert!((left.0 - right.0).abs() < 1e-12);
            assert!((left.1 - right.1).abs() < 1e-12);
        }
    }
}

#[cfg(not(feature = "std"))]
#[path = "support/executor.rs"]
mod executor;
#[cfg(not(feature = "std"))]
use executor::PartitionExecutor;

#[test]
fn additive_exact_factorial_indices_and_paired_bootstrap() {
    let mut a = Vec::new();
    let mut b = Vec::new();
    let mut hybrids = Vec::new();
    for ax in [-1.0, 1.0] {
        for ay in [-1.0, 1.0] {
            for bx in [-1.0, 1.0] {
                for by in [-1.0, 1.0] {
                    a.push(ax + 2.0 * ay);
                    b.push(bx + 2.0 * by);
                    hybrids.extend([bx + 2.0 * ay, ax + 2.0 * by]);
                }
            }
        }
    }
    let evaluations = SobolEvaluations {
        a: &a,
        b: &b,
        hybrids: &hybrids,
        dimensions: 2,
    };
    let indices = sobol_indices(evaluations).unwrap();
    for (observed, expected) in indices.first_order().iter().zip([0.2, 0.8]) {
        assert!((observed - expected).abs() < 1e-12);
    }
    for (observed, expected) in indices.total_order().iter().zip([0.2, 0.8]) {
        assert!((observed - expected).abs() < 1e-12);
    }
    let root = Pcg64::from_seed(558);
    let boot = sobol_bootstrap_scoped(
        evaluations,
        &root,
        100,
        scioxide_stats::ConfidenceProbability::new(0.95).unwrap(),
        scioxide_stats::StatisticsOptions::new().with_max_bytes(10_000),
        ExecutionContext::sequential(),
    )
    .unwrap();
    assert!(boot.first_order()[0].0 < 0.2 && boot.first_order()[0].1 > 0.2);
    assert!(boot.total_order()[1].0 < 0.8 && boot.total_order()[1].1 > 0.8);
    assert!(
        sobol_bootstrap_scoped(
            evaluations,
            &root,
            100,
            scioxide_stats::ConfidenceProbability::new(0.95).unwrap(),
            scioxide_stats::StatisticsOptions::new().with_max_bytes(8),
            ExecutionContext::sequential()
        )
        .is_err()
    );
}

#[test]
fn scrambled_qmc_replicates_recover_additive_indices() {
    let root = Philox4x64::from_seed(558);
    let plan = QmcReplicatePlan::new(
        &QmcPlan::sobol(4).unwrap(),
        &root,
        QmcReplicateOptions::default()
            .with_replicates(4)
            .with_points_per_replicate(256),
    )
    .unwrap();
    let report = sobol_indices_qmc_scoped(
        &plan,
        2,
        256,
        |_, u: f64| Ok(2.0 * u - 1.0),
        |x| Ok(x[0] + 2.0 * x[1]),
        scioxide_stats::StatisticsOptions::new().with_max_bytes(50_000),
        ExecutionContext::sequential(),
    )
    .unwrap();
    assert!((report.first_order()[0] - 0.2).abs() < 0.04);
    assert!((report.total_order()[1] - 0.8).abs() < 0.04);
    assert!(report.randomized_standard_error().is_some());
    #[cfg(feature = "std")]
    let pool = ThreadExecutor::new(NonZeroUsize::new(4).unwrap());
    #[cfg(not(feature = "std"))]
    let pool = PartitionExecutor(NonZeroUsize::new(4).unwrap());
    let replay = sobol_indices_qmc_scoped(
        &plan,
        2,
        256,
        |_, u: f64| Ok(2.0 * u - 1.0),
        |x| Ok(x[0] + 2.0 * x[1]),
        scioxide_stats::StatisticsOptions::new().with_max_bytes(50_000),
        ExecutionContext::new(&pool),
    )
    .unwrap();
    assert_eq!(report, replay);
    assert!(
        sobol_indices_qmc_scoped(
            &plan,
            2,
            128,
            |_, u: f64| Ok(2.0 * u - 1.0),
            |x| Ok(x[0] + 2.0 * x[1]),
            scioxide_stats::StatisticsOptions::new().with_max_bytes(20_000),
            ExecutionContext::sequential(),
        )
        .is_err()
    );
}

#[test]
fn shape_nonfinite_and_degenerate_are_errors() {
    let a = [1.0; 4];
    let e = SobolEvaluations {
        a: &a,
        b: &a,
        hybrids: &a,
        dimensions: 1,
    };
    assert!(sobol_indices(e).is_err());
    let broken = SobolEvaluations {
        hybrids: &a[..3],
        ..e
    };
    assert!(sobol_indices(broken).is_err());
}

#[test]
fn iid_sampler_f32_replays_and_can_bootstrap_evaluations() {
    let root = Philox4x64::from_seed(558);
    let run = || {
        sobol_indices_monte_carlo_scoped(
            2,
            512,
            &root,
            |_, rng: &mut Philox4x64| rng.next_float(),
            |x: &[f32]| Ok(x[0] as f64 + 2.0 * x[1] as f64),
            scioxide_stats::StatisticsOptions::new().with_max_bytes(32_768),
            ExecutionContext::sequential(),
        )
        .unwrap()
    };
    let a = run();
    let b = run();
    assert_eq!(a.indices(), b.indices());
    assert_eq!(a.evaluations().a, b.evaluations().a);
    #[cfg(feature = "std")]
    let pool = ThreadExecutor::new(NonZeroUsize::new(4).unwrap());
    #[cfg(not(feature = "std"))]
    let pool = PartitionExecutor(NonZeroUsize::new(4).unwrap());
    let parallel = sobol_indices_monte_carlo_scoped(
        2,
        512,
        &root,
        |_, rng: &mut Philox4x64| rng.next_float(),
        |x: &[f32]| Ok(x[0] as f64 + 2.0 * x[1] as f64),
        scioxide_stats::StatisticsOptions::new().with_max_bytes(32_768),
        ExecutionContext::new(&pool),
    )
    .unwrap();
    assert_eq!(a.indices(), parallel.indices());
    assert_eq!(a.evaluations().hybrids, parallel.evaluations().hybrids);
    assert!(
        (a.indices().first_order()[0] - 0.2).abs() < 0.12,
        "{:?}",
        a.indices().first_order()
    );
    let bootstrap = sobol_bootstrap_scoped(
        a.evaluations(),
        &Pcg64::from_seed(559),
        32,
        scioxide_stats::ConfidenceProbability::new(0.9).unwrap(),
        scioxide_stats::StatisticsOptions::new().with_max_bytes(20_000),
        ExecutionContext::sequential(),
    )
    .unwrap();
    assert_eq!(bootstrap.first_order().len(), 2);
}
