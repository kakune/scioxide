#![allow(clippy::unwrap_used)]
use scioxide_core::{
    ExecutionContext, RealScalar, SciOxideError, SciOxideErrorKind, SciOxideErrorView,
};
use scioxide_random::*;
/// Asserts one point per stratum in each of the `dim` columns of an `n`-point design.
fn assert_latin<S: RealScalar>(design: &[S], n: usize, dim: usize) {
    for col in 0..dim {
        let mut bins: Vec<_> = design
            .chunks_exact(dim)
            .map(|r| (r[col].into_f64() * n as f64) as usize)
            .collect();
        bins.sort();
        assert_eq!(bins, (0..n).collect::<Vec<_>>());
    }
}
fn check<S: RealScalar>() {
    let options = LatinHypercubeOptions::default().with_max_proposals(40);
    let mut out = vec![S::ZERO; 48];
    let mut work = SamplingWorkspace::new();
    let report = latin_hypercube_into(
        3,
        &mut SplitMix64::from_seed(42),
        Rollback,
        options,
        &mut out,
        &mut work,
        ExecutionContext::sequential_native(),
    )
    .unwrap();
    assert!(report.final_discrepancy().unwrap() <= report.initial_discrepancy().unwrap());
    assert!(report.proposals() <= 40);
    assert_latin(&out, 16, 3);
    let (again, repeated) =
        latin_hypercube_with_options::<S, _>(16, 3, &mut SplitMix64::from_seed(42), options)
            .unwrap();
    assert_eq!(out, again);
    assert_eq!(repeated.proposals(), report.proposals());
    assert_eq!(repeated.final_discrepancy(), report.final_discrepancy());
    // The one-shot equals the configured and caller-buffer forms with the defaults.
    let plain = latin_hypercube::<S, _>(16, 3, &mut SplitMix64::from_seed(7)).unwrap();
    let (configured, report) = latin_hypercube_with_options::<S, _>(
        16,
        3,
        &mut SplitMix64::from_seed(7),
        LatinHypercubeOptions::default(),
    )
    .unwrap();
    let mut buffered = vec![S::ZERO; 48];
    latin_hypercube_into(
        3,
        &mut SplitMix64::from_seed(7),
        AdvanceOnError,
        LatinHypercubeOptions::default(),
        &mut buffered,
        &mut work,
        ExecutionContext::sequential_native(),
    )
    .unwrap();
    let bits = |x: &[S]| x.iter().map(|v| v.into_f64().to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&plain), bits(&configured));
    assert_eq!(bits(&plain), bits(&buffered));
    assert_eq!(report.termination(), LatinHypercubeTermination::Complete);
    assert_latin(&plain, 16, 3);
    let opts = PoissonDiskOptions::default()
        .with_candidates_per_point(30)
        .with_max_proposals(3000)
        .with_max_grid_cells(1000)
        .with_max_distance_checks(100_000);
    let plan = PoissonDisk::with_options(
        &[S::ZERO; 2],
        &[S::ONE; 2],
        S::try_from_f64_lossy(0.12).unwrap(),
        40,
        opts,
    )
    .unwrap();
    let mut out = vec![S::ZERO; plan.output_elements()];
    let mut work = plan.workspace().unwrap();
    let report = plan
        .generate_into(
            &mut SplitMix64::from_seed(3),
            Rollback,
            &mut out,
            &mut work,
            ExecutionContext::sequential_native(),
        )
        .unwrap();
    assert!(report.points() > 15);
    assert!(report.proposals() <= 3000);
    assert!(report.distance_checks() <= 100_000);
    for i in 0..report.points() {
        for j in 0..i {
            let r = out[2 * i..2 * i + 2]
                .iter()
                .zip(&out[2 * j..2 * j + 2])
                .map(|(a, b)| (a.into_f64() - b.into_f64()).powi(2))
                .sum::<f64>()
                .sqrt();
            assert!(r >= S::try_from_f64_lossy(0.12).unwrap().into_f64());
        }
    }
    let saved = out.clone();
    let repeat = plan
        .generate_into(
            &mut SplitMix64::from_seed(3),
            Rollback,
            &mut out,
            &mut work,
            ExecutionContext::sequential_native(),
        )
        .unwrap();
    assert_eq!(repeat.points(), report.points());
    assert_eq!(out, saved);
    let (owned, owned_report) = plan.generate(&mut SplitMix64::from_seed(3)).unwrap();
    assert_eq!(owned, &saved[..2 * report.points()]);
    assert_eq!(owned_report.termination(), report.termination());
    assert_eq!(owned_report.proposals(), report.proposals());
}
#[test]
fn designs_both_widths() {
    check::<f32>();
    check::<f64>();
}
#[test]
fn budgets_and_validation() {
    let opts = PoissonDiskOptions::default()
        .with_candidates_per_point(30)
        .with_max_proposals(1)
        .with_max_grid_cells(1000)
        .with_max_distance_checks(0);
    let plan = PoissonDisk::with_options(&[0.; 2], &[1.; 2], 0.2, 50, opts).unwrap();
    let mut out = vec![-1.; 100];
    let report = plan
        .generate_into(
            &mut SplitMix64::from_seed(0),
            Rollback,
            &mut out,
            &mut plan.workspace().unwrap(),
            ExecutionContext::sequential_native(),
        )
        .unwrap();
    assert_eq!(report.points(), 1);
    assert_eq!(
        report.termination(),
        PoissonDiskTermination::BudgetExhausted
    );
    assert!(out[2..].iter().all(|&x| x == -1.));
    assert!(PoissonDisk::with_options(&[0.; 2], &[1.; 2], 1e-20, 50, opts).is_err());
    assert!(PoissonDisk::with_options(&[0.; 2], &[1.; 2], f64::NAN, 50, opts).is_err());
    let invalid = |error: SciOxideError, expected: &str| {
        assert!(
            matches!(error.view(), SciOxideErrorView::InvalidInput { argument, .. }
                if argument.as_str() == expected),
            "expected {expected}: {error}"
        );
    };
    invalid(
        PoissonDisk::new(&[0.; 2], &[1.; 2], 0.2, 0).unwrap_err(),
        "capacity",
    );
    let mut rng = SplitMix64::from_seed(0);
    invalid(latin_hypercube::<f64, _>(0, 2, &mut rng).unwrap_err(), "n");
    invalid(
        latin_hypercube::<f64, _>(2, 0, &mut rng).unwrap_err(),
        "dim",
    );
    let into = |dim, out: &mut [f64], options| {
        latin_hypercube_into(
            dim,
            &mut SplitMix64::from_seed(0),
            AdvanceOnError,
            options,
            out,
            &mut SamplingWorkspace::new(),
            ExecutionContext::sequential_native(),
        )
        .unwrap_err()
    };
    let default = LatinHypercubeOptions::default();
    invalid(into(2, &mut [], default), "out");
    invalid(
        into(2, &mut [0.; 4], default.with_stagnation_limit(0)),
        "options.stagnation_limit",
    );
    assert_eq!(
        into(2, &mut [0.; 5], default).kind(),
        SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(rng.position(), Some(0));
}

#[test]
fn poisson_disk_one_shot_equals_generate_and_maps_work_limits() {
    let points = poisson_disk(
        &[0.0; 2],
        &[1.0; 2],
        0.1,
        64,
        &mut SplitMix64::from_seed(11),
    )
    .unwrap();
    let (generated, report) = PoissonDisk::new(&[0.0; 2], &[1.0; 2], 0.1, 64)
        .unwrap()
        .generate(&mut SplitMix64::from_seed(11))
        .unwrap();
    assert_eq!(points, generated);
    assert_eq!(points.len(), 2 * report.points());
    assert!(report.points() <= 64);
    let separation =
        geometric_discrepancy(&points, 2, GeometricDiscrepancy::MinimumDistance).unwrap();
    assert!(separation >= 0.1, "{separation}");
    // A radius far below the box exhausts the default limit of 2²⁰ candidates before the
    // capacity is reached: every retired point costs 30 failed candidates.
    let error = poisson_disk(
        &[0.0],
        &[1.0],
        1e-6,
        1_000_000,
        &mut SplitMix64::from_seed(11),
    )
    .unwrap_err();
    assert!(
        matches!(
            error.view(),
            SciOxideErrorView::ConvergenceFailure {
                reason: scioxide_core::ConvergenceReason::IterationLimit,
                ..
            }
        ),
        "{error}"
    );
}
