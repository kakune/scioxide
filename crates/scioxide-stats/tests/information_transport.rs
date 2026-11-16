#![allow(clippy::unwrap_used, clippy::panic)]
use scioxide_core::*;
use scioxide_stats::*;
fn identity() -> SolveIdentity {
    SolveIdentity::new(
        ProblemToken::new(552),
        scioxide_core::algorithm_id!("stats", "test", 1),
        0,
        0,
    )
}
fn finite_value<S: RealScalar>(v: InformationValue<S>) -> f64 {
    match v {
        InformationValue::Finite(v) => v.try_to_f64_exact().unwrap(),
        other => panic!("unexpected {other:?}"),
    }
}

fn probability_fixtures<S: RealScalar>(scalar: &str) {
    let parse = |bits: &str| {
        let bits = u64::from_str_radix(bits, 16).unwrap();
        let value = if scalar == "f32" {
            f32::from_bits(bits as u32) as f64
        } else {
            f64::from_bits(bits)
        };
        S::try_from_f64_lossy(value).unwrap()
    };
    let mut cases = 0;
    for line in include_str!("data/information.tsv").lines() {
        if line.starts_with('#') {
            continue;
        }
        let row: Vec<_> = line.split('\t').collect();
        if row[0] != scalar {
            continue;
        }
        cases += 1;
        let p: Vec<_> = row[4].split(',').map(parse).collect();
        let q: Option<Vec<_>> = (row[5] != "-").then(|| row[5].split(',').map(parse).collect());
        let value = entropy_with_options(
            &p,
            q.as_deref(),
            EntropyOptions::new()
                .with_base(parse(row[3]))
                .with_normalization(MassNormalization::Normalize),
        )
        .unwrap();
        if row[6] == "inf" {
            assert_eq!(value, InformationValue::PositiveInfinity);
            continue;
        }
        let expected = parse(row[6]).into_f64();
        let actual = finite_value(value);
        let condition: f64 = row[7].parse().unwrap();
        // Destination rounding plus compensated binary64 arithmetic, scaled
        // by the sum of absolute terms for cancellation-sensitive KL values.
        let tolerance = 2.0 * S::EPSILON.into_f64() * expected.abs()
            + 64.0 * f64::EPSILON * condition
            + 4.0 * S::MIN_SUBNORMAL.into_f64();
        assert!(
            (actual - expected).abs() <= tolerance,
            "{scalar} {} {} base={:?}: {actual:e} != {expected:e} (tolerance {tolerance:e})",
            row[1],
            row[2],
            parse(row[3]),
        );
    }
    assert!(cases >= 96);
}

#[test]
fn independent_probability_information() {
    probability_fixtures::<f32>("f32");
    probability_fixtures::<f64>("f64");
}

fn probability_admission<S: RealScalar>() {
    let s = |x| S::try_from_f64_lossy(x).unwrap();
    let options = EntropyOptions::<S>::default();
    for p in [
        &[][..],
        &[S::ZERO, S::ZERO][..],
        &[S::ONE, -S::ONE][..],
        &[S::ONE, S::NAN][..],
        &[S::ONE, S::INFINITY][..],
    ] {
        assert!(entropy_with_options(p, None, options).is_err());
    }
    for base in [S::ZERO, S::ONE, -S::ONE, S::NAN, S::INFINITY] {
        assert!(entropy_with_options(&[S::ONE], None, options.with_base(base)).is_err());
    }
    for tolerance in [-S::ONE, S::NAN, S::INFINITY] {
        assert!(
            entropy_with_options(
                &[S::ONE],
                None,
                options.with_normalization(MassNormalization::RequireUnit(tolerance))
            )
            .is_err()
        );
    }
    let p = [s(0.5), s(0.5) + S::EPSILON];
    for (tolerance, accepted) in [(S::EPSILON, true), (S::EPSILON * s(0.5), false)] {
        let options = options.with_normalization(MassNormalization::RequireUnit(tolerance));
        assert_eq!(entropy_with_options(&p, None, options).is_ok(), accepted);
        assert_eq!(
            entropy_with_options(&[s(0.5); 2], Some(&p), options).is_ok(),
            accepted
        );
    }
    let unit = options.with_normalization(MassNormalization::RequireUnit(S::ZERO));
    assert!(entropy_with_options(&[s(0.25), s(0.75)], None, unit).is_ok());
    assert!(entropy_with_options(&[S::ONE, S::ONE], None, unit).is_err());
    assert!(entropy_with_options(&[S::ONE, S::ONE], Some(&[S::ONE]), options).is_err());
    // Entire supports must be admitted before returning mathematical infinity.
    for q in [
        [S::ZERO, S::ZERO],
        [S::ZERO, S::NAN],
        [S::ZERO, -S::ONE],
        [S::ZERO, S::INFINITY],
    ] {
        assert!(entropy_with_options(&[S::ONE, S::ONE], Some(&q), options).is_err());
    }
    assert_eq!(
        entropy_with_options(&[S::ONE, S::ZERO], Some(&[S::ZERO, S::ONE]), options).unwrap(),
        InformationValue::PositiveInfinity
    );
}

#[test]
fn probability_admission_and_normalization_boundaries() {
    probability_admission::<f32>();
    probability_admission::<f64>();
    let p = vec![1.0_f64; 65_536];
    let value = finite_value(entropy_with_options(&p, None, EntropyOptions::default()).unwrap());
    assert!((value - (p.len() as f64).ln()).abs() <= 16.0 * f64::EPSILON * value);
}

fn near_identical_kl<S: RealScalar>() {
    let options = EntropyOptions::<S>::default();
    let delta = S::EPSILON;
    let value = finite_value(
        entropy_with_options(
            &[S::ONE + delta, S::ONE - delta],
            Some(&[S::ONE; 2]),
            options,
        )
        .unwrap(),
    );
    // For the symmetric binary perturbation, KL = delta^2/2 + O(delta^4).
    // The next term is below destination precision for either scalar width.
    let expected = delta.into_f64().powi(2) / (2.0 * options.base().into_f64().ln());
    assert!(value > 0.0);
    assert!((value / expected - 1.0).abs() <= 8.0 * S::EPSILON.into_f64());
}

#[test]
fn probability_kl_retains_ulp_scale_perturbations() {
    near_identical_kl::<f32>();
    near_identical_kl::<f64>();
}

fn information<S: RealScalar>() {
    let s = |x| S::try_from_f64_lossy(x).unwrap();
    let opts = EntropyOptions::new()
        .with_base(s(2.))
        .with_normalization(MassNormalization::Normalize);
    let h = finite_value(entropy_with_options(&[s(1.), s(1.), s(0.)], None, opts).unwrap());
    assert!((h - 1.).abs() < 2e-6);
    assert_eq!(
        entropy_with_options(&[s(1.), s(0.)], Some(&[s(0.), s(1.)]), opts).unwrap(),
        InformationValue::PositiveInfinity
    );
    assert!(
        finite_value(entropy_with_options(&[s(1.), s(3.)], Some(&[s(2.), s(6.)]), opts).unwrap())
            .abs()
            < 2e-6
    );
    for method in [
        DifferentialEntropyMethod::Vasicek,
        DifferentialEntropyMethod::VanEs,
        DifferentialEntropyMethod::Ebrahimi,
        DifferentialEntropyMethod::Correa,
    ] {
        let options = DifferentialEntropyOptions::new()
            .with_method(method)
            .with_window(Some(1))
            .with_base(s(core::f64::consts::E));
        let x = [s(0.), s(1.), s(2.), s(3.), s(4.)];
        let y = [s(7.), s(9.), s(11.), s(13.), s(15.)];
        let h = finite_value(differential_entropy_with_options(&x, options).unwrap());
        let shifted = finite_value(differential_entropy_with_options(&y, options).unwrap());
        assert!(
            (shifted - h - 2f64.ln()).abs() < 3e-6,
            "{method:?}: {h} {shifted}"
        );
        // Independent exact spacings for equally-spaced n=5,m=1.
        let expected = match method {
            DifferentialEntropyMethod::Vasicek => (2. * 2.5f64.ln() + 3. * 5f64.ln()) / 5.,
            DifferentialEntropyMethod::VanEs => 137. / 60.,
            DifferentialEntropyMethod::Ebrahimi => 5f64.ln(),
            DifferentialEntropyMethod::Correa => (2. * (10f64 / 3.).ln() + 3. * 5f64.ln()) / 5.,
            _ => unreachable!(),
        };
        assert!((h - expected).abs() < 3e-6, "{method:?}: {h} {expected}");
        assert_eq!(
            differential_entropy_with_options(&[s(1.); 5], options).unwrap(),
            InformationValue::NegativeInfinity
        );
    }
    let left = [s(0.), s(2.)];
    let right = [s(1.), s(3.)];
    let w = distribution_distance_with_options(
        &left,
        Unweighted,
        &right,
        Unweighted,
        DistributionDistance::Wasserstein,
        StatisticsOptions::new(),
    )
    .unwrap();
    assert!((w.try_to_f64_exact().unwrap() - 1.).abs() < 2e-6);
    let e = distribution_distance_with_options(
        &left,
        Unweighted,
        &right,
        Unweighted,
        DistributionDistance::Energy,
        StatisticsOptions::new(),
    )
    .unwrap();
    assert!((e.try_to_f64_exact().unwrap() - 1.).abs() < 2e-6);
    assert_eq!(
        distribution_distance_with_options(
            &left,
            Unweighted,
            &left,
            Unweighted,
            DistributionDistance::Wasserstein,
            StatisticsOptions::new()
        )
        .unwrap(),
        s(0.)
    );
}
#[test]
fn exact_information_and_distances() {
    information::<f32>();
    information::<f64>();
}
struct TwoPoint {
    bad: bool,
    calls: usize,
}
impl<S: RealScalar> TransportSolver<S> for TwoPoint {
    fn solve(
        &mut self,
        p: TransportProblem<'_, S>,
        c: TransportCandidate<'_, S>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<StatsTermination, SciOxideError> {
        self.calls += 1;
        // Independent exact diagonal transport: translated two-point measures.
        c.coupling.fill(S::ZERO);
        c.coupling[0] = p.source[0];
        c.coupling[3] = p.source[1];
        c.source_dual.copy_from_slice(&[p.costs[0], p.costs[3]]);
        c.target_dual.fill(S::ZERO);
        if self.bad {
            c.coupling[0] = S::ZERO;
        }
        Ok(StatsTermination::Converged)
    }
}
fn transport<S: RealScalar>() {
    let s = |x| S::try_from_f64_lossy(x).unwrap();
    let left = [s(0.), s(0.), s(0.), s(2.)];
    let right = [s(1.), s(0.), s(1.), s(2.)];
    let shape = [2, 2];
    let strides = [2, 1];
    let layout = StridedLayout::new(&shape, &strides).unwrap();
    let a = BatchView::new(&left, layout, Axis::Index(0))
        .and_then(|view| view.with_coordinates(Axis::Index(1)))
        .unwrap();
    let b = BatchView::new(&right, layout, Axis::Index(0))
        .and_then(|view| view.with_coordinates(Axis::Index(1)))
        .unwrap();
    let options = TransportOptions::new()
        .with_max_pairs(4)
        .with_max_workspace_bytes(4096);
    let mut w = TransportWorkspace::default();
    w.reserve(2, 2, options).unwrap();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut ctx = state
        .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
        .unwrap();
    let mut solver = TwoPoint {
        bad: false,
        calls: 0,
    };
    let result = scioxide_stats::wasserstein_distance_nd_scoped(
        a,
        b,
        StatisticsOptions::new(),
        options,
        &mut solver,
        &mut w,
        &mut ctx,
    )
    .unwrap();
    assert_eq!(result.termination, StatsTermination::Converged);
    assert!((result.distance.unwrap().try_to_f64_exact().unwrap() - 1.).abs() < 2e-6);
    solver.bad = true;
    assert!(
        scioxide_stats::wasserstein_distance_nd_scoped(
            a,
            b,
            StatisticsOptions::new(),
            options,
            &mut solver,
            &mut w,
            &mut ctx
        )
        .is_err()
    );
    let limited = options.with_max_pairs(3);
    assert!(
        scioxide_stats::wasserstein_distance_nd_scoped(
            a,
            b,
            StatisticsOptions::new(),
            limited,
            &mut solver,
            &mut w,
            &mut ctx
        )
        .is_err()
    );
    assert_eq!(solver.calls, 2);
}
#[test]
fn verified_injected_transport() {
    transport::<f32>();
    transport::<f64>();
}
#[test]
fn sigma_clipping_budgets_and_indices() {
    let mut w = StatisticsWorkspace::default();
    w.try_reserve(&StatisticsWorkspace::scalar_requirements(6, StatisticsOptions::new()).unwrap())
        .unwrap();
    let data = [0f64, 0., 0., 0., 0., 10.];
    let mut out = [99.; 6];
    let mut indices = [99; 6];
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut ctx = state
        .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
        .unwrap();
    let options = SigmaClipOptions::new().with_max_iterations(8);
    let r = scioxide_stats::sigma_clip_into(
        &data,
        2.,
        2.,
        options,
        &mut out,
        &mut indices,
        &mut w,
        &mut ctx,
    )
    .unwrap();
    assert_eq!(r.termination(), StatsTermination::Converged);
    assert_eq!(r.retained(), 5);
    assert_eq!(&indices[..5], &[0, 1, 2, 3, 4]);
    let r = scioxide_stats::sigma_clip_into(
        &data,
        2.0,
        2.0,
        options.with_max_iterations(0),
        &mut out,
        &mut indices,
        &mut w,
        &mut ctx,
    )
    .unwrap();
    assert_eq!(r.termination(), StatsTermination::BudgetExhausted);
    assert_eq!(r.retained(), 6);
}
#[test]
fn provider_admission_and_dual_verification() {
    let shape = [2, 2];
    let strides = [2, 1];
    let layout = StridedLayout::new(&shape, &strides).unwrap();
    let a = BatchView::new(&[0., 0., 0., 2.], layout, Axis::Index(0))
        .and_then(|view| view.with_coordinates(Axis::Index(1)))
        .unwrap();
    let b = BatchView::new(&[1., 0., 1., 2.], layout, Axis::Index(0))
        .and_then(|view| view.with_coordinates(Axis::Index(1)))
        .unwrap();
    let options = TransportOptions::new()
        .with_max_pairs(4)
        .with_max_workspace_bytes(4096);
    let mut workspace = TransportWorkspace::default();
    workspace.reserve(2, 2, options).unwrap();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, Budget::ONE);
    let mut ctx = state
        .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
        .unwrap();
    let mut solver = TwoPoint {
        bad: false,
        calls: 0,
    };
    scioxide_stats::wasserstein_distance_nd_scoped(
        a,
        b,
        StatisticsOptions::new(),
        options,
        &mut solver,
        &mut workspace,
        &mut ctx,
    )
    .unwrap();
    let r = scioxide_stats::wasserstein_distance_nd_scoped(
        a,
        b,
        StatisticsOptions::new(),
        options,
        &mut solver,
        &mut workspace,
        &mut ctx,
    )
    .unwrap();
    assert_eq!(r.termination, StatsTermination::BudgetExhausted);
    assert!(r.distance.is_none());
    assert_eq!(solver.calls, 1);
    struct BadDual;
    impl TransportSolver<f64> for BadDual {
        fn solve(
            &mut self,
            p: TransportProblem<'_, f64>,
            c: TransportCandidate<'_, f64>,
            _: &mut SolveContext<'_, '_>,
        ) -> Result<StatsTermination, SciOxideError> {
            c.coupling.copy_from_slice(&[0.5, 0., 0., 0.5]);
            c.source_dual.fill(10. * p.cost_scale);
            c.target_dual.fill(0.);
            Ok(StatsTermination::Converged)
        }
    }
    let mut budgets = SolveBudgets::unbounded();
    let mut ctx = state
        .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
        .unwrap();
    assert!(
        scioxide_stats::wasserstein_distance_nd_scoped(
            a,
            b,
            StatisticsOptions::new(),
            options,
            &mut BadDual,
            &mut workspace,
            &mut ctx
        )
        .is_err()
    );
}
#[test]
fn weighted_distance_and_information_extremes() {
    let options = EntropyOptions::<f64>::default();
    assert!(
        (finite_value(entropy_with_options(&[f64::MAX, f64::MAX], None, options).unwrap())
            - 2f64.ln())
        .abs()
            < 1e-14
    );
    assert!(entropy_with_options(&[0., 0.], None, options).is_err());
    assert!(
        entropy_with_options(
            &[1., 1.],
            None,
            options.with_normalization(MassNormalization::RequireUnit(1e-12))
        )
        .is_err()
    );
    let x = [0., 4.];
    let w = [3u64, 1];
    let y = [1., 2.];
    let result = distribution_distance_with_options(
        &x,
        Weights::<_, Frequency>::new(&w),
        &y,
        Unweighted,
        DistributionDistance::Wasserstein,
        StatisticsOptions::new(),
    )
    .unwrap();
    let expanded = distribution_distance_with_options(
        &[0., 0., 0., 4.],
        Unweighted,
        &y,
        Unweighted,
        DistributionDistance::Wasserstein,
        StatisticsOptions::new(),
    )
    .unwrap();
    assert_eq!(result, expanded);
    for metric in [
        DistributionDistance::Wasserstein,
        DistributionDistance::Energy,
    ] {
        let a = distribution_distance_with_options(
            &x,
            Unweighted,
            &y,
            Unweighted,
            metric,
            StatisticsOptions::new(),
        )
        .unwrap();
        let b = distribution_distance_with_options(
            &y,
            Unweighted,
            &x,
            Unweighted,
            metric,
            StatisticsOptions::new(),
        )
        .unwrap();
        assert!((a - b).abs() < 1e-14);
        let shifted = distribution_distance_with_options(
            &[7., 15.],
            Unweighted,
            &[9., 11.],
            Unweighted,
            metric,
            StatisticsOptions::new(),
        )
        .unwrap();
        let factor = if metric == DistributionDistance::Wasserstein {
            2.
        } else {
            2f64.sqrt()
        };
        assert!((shifted - factor * a).abs() < 1e-14);
    }
}
#[test]
fn unequal_mass_transport_matches_independent_corner_optimum() {
    struct Corner;
    impl TransportSolver<f64> for Corner {
        fn solve(
            &mut self,
            _: TransportProblem<'_, f64>,
            c: TransportCandidate<'_, f64>,
            _: &mut SolveContext<'_, '_>,
        ) -> Result<StatsTermination, SciOxideError> {
            c.coupling.copy_from_slice(&[0.25, 0.5, 0., 0.25]);
            c.source_dual.copy_from_slice(&[0.25, -0.25]);
            c.target_dual.copy_from_slice(&[0., 0.5]);
            Ok(StatsTermination::Converged)
        }
    }
    // All feasible 2x2 couplings are (t, .75-t, .25-t, t), 0<=t<=.25.
    // Original-coordinate cost is 3-4t, so the independently enumerated minimum is 2.
    let shape = [2, 2];
    let strides = [2, 1];
    let layout = StridedLayout::new(&shape, &strides).unwrap();
    let a = BatchView::new(&[0., 0., 4., 0.], layout, Axis::Index(0))
        .and_then(|view| view.with_coordinates(Axis::Index(1)))
        .unwrap()
        .with_weights(Weights::<_, Mass>::new(&[3., 1.]))
        .unwrap();
    let b = BatchView::new(&[1., 0., 3., 0.], layout, Axis::Index(0))
        .and_then(|view| view.with_coordinates(Axis::Index(1)))
        .unwrap()
        .with_weights(Weights::<_, Frequency>::new(&[1, 3]))
        .unwrap();
    let options = TransportOptions::new()
        .with_max_pairs(4)
        .with_max_workspace_bytes(4096);
    let mut w = TransportWorkspace::default();
    w.reserve(2, 2, options).unwrap();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
        .unwrap();
    let result = scioxide_stats::wasserstein_distance_nd_scoped(
        a,
        b,
        StatisticsOptions::new(),
        options,
        &mut Corner,
        &mut w,
        &mut context,
    )
    .unwrap();
    assert_eq!(result.distance, Some(2.));
    assert_eq!(result.normalized_duality_gap, Some(0.));
}
#[test]
fn transport_checks_aggregate_missing_mass() {
    struct Empty;
    impl TransportSolver<f64> for Empty {
        fn solve(
            &mut self,
            _: TransportProblem<'_, f64>,
            _: TransportCandidate<'_, f64>,
            _: &mut SolveContext<'_, '_>,
        ) -> Result<StatsTermination, SciOxideError> {
            Ok(StatsTermination::Converged)
        }
    }
    // Per-entry residual .5 would satisfy this deliberately loose .6 absolute
    // threshold, but the full missing unit of mass must fail the L1 contract.
    let layout = StridedLayout::new(&[2], &[1]).unwrap();
    let a = BatchView::new(&[0., 0.], layout, Axis::Index(0)).unwrap();
    let b = BatchView::new(&[1., 1.], layout, Axis::Index(0)).unwrap();
    let options = TransportOptions::new()
        .with_max_pairs(4)
        .with_max_workspace_bytes(4096)
        .with_feasibility(FeasibilityTolerance::new(0.6, 0.).unwrap());
    let mut w = TransportWorkspace::default();
    w.reserve(2, 2, options).unwrap();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut ctx = state
        .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
        .unwrap();
    assert!(
        scioxide_stats::wasserstein_distance_nd_scoped(
            a,
            b,
            StatisticsOptions::new(),
            options,
            &mut Empty,
            &mut w,
            &mut ctx
        )
        .is_err()
    );
}
