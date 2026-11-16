//! Shared contracts of the fixed one-dimensional rules: Gaussian plans (unweighted
//! and weighted), Gauss–Kronrod, Clenshaw–Curtis and composite Newton–Cotes.
#![allow(clippy::unwrap_used)]

use core::fmt::Debug;
use std::sync::atomic::{AtomicUsize, Ordering};

use scioxide_core::{
    AffineValueMut, AttemptOutcome, Budget, Complex, ExecutionContext, Interval, NonFiniteOrigin,
    NumericScalar, NumericalIssue, RealScalar, SciOxideError, SciOxideErrorView, SolveScope,
};
use scioxide_integrate::{
    ClenshawCurtis, FixedRule, FixedRuleNode, FixedRuleWorkspace, GaussKronrod, GaussKronrodRule,
    GaussianRuleOptions, GaussianRulePlan, GaussianRuleVariant, GaussianSupport,
    IntegrationTermination, NewtonCotes, NewtonCotesEndpoints,
};
use scioxide_poly::OrthogonalFamily;

fn native<'scope>() -> SolveScope<'scope, 'static> {
    ExecutionContext::sequential_native().into()
}

fn owned_and_degenerate<S, R>(rule: &R)
where
    S: RealScalar + Default + Debug,
    R: FixedRule<S>,
{
    let interval = Interval::new(S::ZERO, S::TWO).unwrap();
    let (owned, report) = rule.integrate(|x: S| Ok(x * x), interval).unwrap();
    let mut workspace = rule.workspace(1).unwrap();
    let mut out = S::ZERO;
    let prepared = rule
        .integrate_into(
            |x: S| Ok(x * x),
            interval,
            &mut out,
            &mut workspace,
            native(),
        )
        .unwrap();
    assert_eq!(owned, out);
    assert_eq!(report, prepared);

    let calls = AtomicUsize::new(0);
    let counted = |x: S| {
        calls.fetch_add(1, Ordering::Relaxed);
        Ok(x)
    };
    let point = Interval::new(S::ONE, S::ONE).unwrap();
    let (value, report) = rule.integrate(&counted, point).unwrap();
    assert_eq!(value, S::ZERO);
    assert_eq!(
        report.termination(),
        IntegrationTermination::DegenerateDomain
    );
    assert_eq!(report.evaluations(), 0);

    // Adjacent representable bounds collapse distinct canonical nodes.
    let tiny = Interval::new(S::ONE, S::ONE.next_after(S::INFINITY)).unwrap();
    let error = rule.integrate(&counted, tiny).unwrap_err();
    assert!(matches!(
        error.view(),
        SciOxideErrorView::NumericalInstability {
            issue: NumericalIssue::Breakdown,
            ..
        }
    ));
    assert_eq!(calls.load(Ordering::Relaxed), 0);
}

fn degenerate_cases<S>()
where
    S: RealScalar + NumericScalar<Real = S> + Default + Debug,
{
    owned_and_degenerate(&GaussianRulePlan::<S>::gauss_legendre(4).unwrap());
    owned_and_degenerate::<S, _>(&GaussKronrod::new(GaussKronrodRule::G7K15).unwrap());
    owned_and_degenerate(&ClenshawCurtis::<S>::new(6).unwrap());
    owned_and_degenerate(
        &NewtonCotes::<S>::new(
            3,
            NewtonCotesEndpoints::Closed,
            Budget::try_from(2).unwrap(),
        )
        .unwrap(),
    );
}

#[test]
fn owned_forms_match_prepared_forms_and_geometry_is_checked_first() {
    degenerate_cases::<f64>();
    degenerate_cases::<f32>();
}

/// A rule whose mapped metadata leaves the interval.
struct Escaping;

impl FixedRule<f64> for Escaping {
    fn algorithm_id(&self) -> scioxide_core::AlgorithmId {
        scioxide_core::algorithm_id!("integrate", "Escaping", 1)
    }

    fn node_count(&self) -> usize {
        2
    }

    fn map_nodes_into(
        &self,
        interval: Interval<f64>,
        destination: &mut [FixedRuleNode<f64>],
    ) -> Result<(), SciOxideError> {
        destination[0] = FixedRuleNode::new(interval.lower() - 1.0, 1.0, None);
        destination[1] = FixedRuleNode::new(interval.upper(), 1.0, None);
        Ok(())
    }
}

#[test]
fn out_of_domain_metadata_is_a_breakdown_before_callbacks() {
    let calls = AtomicUsize::new(0);
    let error = Escaping
        .integrate(
            |x: f64| {
                calls.fetch_add(1, Ordering::Relaxed);
                Ok(x)
            },
            Interval::new(0.0, 1.0).unwrap(),
        )
        .unwrap_err();
    assert!(matches!(
        error.view(),
        SciOxideErrorView::NumericalInstability {
            issue: NumericalIssue::Breakdown,
            ..
        }
    ));
    assert_eq!(calls.load(Ordering::Relaxed), 0);
}

#[test]
fn weighted_supports_are_validated_before_any_callback() {
    let calls = AtomicUsize::new(0);
    let counted = |x: f64| {
        calls.fetch_add(1, Ordering::Relaxed);
        Ok(x)
    };
    let laguerre = GaussianRulePlan::<f64>::new(OrthogonalFamily::Laguerre, 4).unwrap();
    let hermite = GaussianRulePlan::<f64>::new(OrthogonalFamily::HermiteHe, 4).unwrap();
    let chebyshev = GaussianRulePlan::<f64>::new(OrthogonalFamily::ChebyshevT, 4).unwrap();
    let mut workspace = laguerre.workspace(1).unwrap();
    let mut out = 3.0;
    let unit = Interval::new(0.0, 1.0).unwrap();
    for (plan, support) in [
        (&laguerre, GaussianSupport::RealLine),
        (&laguerre, GaussianSupport::Interval(unit)),
        (&hermite, GaussianSupport::HalfLine(0.0)),
        (&chebyshev, GaussianSupport::RealLine),
    ] {
        let error = plan
            .integrate_weighted_into(&counted, support, &mut out, &mut workspace, native())
            .unwrap_err();
        assert!(matches!(
            error.view(),
            SciOxideErrorView::InvalidInput { argument, .. }
                if argument == scioxide_core::argument!("support")
        ));
        assert_eq!(
            workspace.last_attempt().unwrap().outcome(),
            AttemptOutcome::PreflightRejected
        );
    }
    let error = laguerre
        .integrate_weighted_into(
            &counted,
            GaussianSupport::HalfLine(f64::NAN),
            &mut out,
            &mut workspace,
            native(),
        )
        .unwrap_err();
    assert!(matches!(
        error.view(),
        SciOxideErrorView::NonFinite {
            origin: NonFiniteOrigin::Input,
            ..
        }
    ));
    // Weighted families never silently integrate an unweighted `∫ f dx`.
    for plan in [&laguerre, &hermite, &chebyshev] {
        let error = plan
            .integrate_into(&counted, unit, &mut out, &mut workspace, native())
            .unwrap_err();
        assert!(matches!(
            error.view(),
            SciOxideErrorView::InvalidInput { .. }
        ));
        assert_eq!(
            workspace.last_attempt().unwrap().outcome(),
            AttemptOutcome::PreflightRejected
        );
        assert!(matches!(
            plan.map_nodes_into(unit, &mut [FixedRuleNode::new(0.0, 0.0, None); 8])
                .unwrap_err()
                .view(),
            SciOxideErrorView::InvalidInput { .. }
        ));
    }
    // A huge half-line shift collapses the Laguerre nodes onto the bound.
    let error = laguerre
        .integrate_weighted_into(
            &counted,
            GaussianSupport::HalfLine(1.0e30),
            &mut out,
            &mut workspace,
            native(),
        )
        .unwrap_err();
    assert!(matches!(
        error.view(),
        SciOxideErrorView::NumericalInstability {
            issue: NumericalIssue::Breakdown,
            ..
        }
    ));
    // A zero-width weighted interval is the exact degenerate zero.
    let (value, report) = chebyshev
        .integrate_weighted(
            &counted,
            GaussianSupport::Interval(Interval::new(2.0, 2.0).unwrap()),
        )
        .unwrap();
    assert_eq!(value, 0.0);
    assert_eq!(
        report.termination(),
        IntegrationTermination::DegenerateDomain
    );
    assert_eq!(out, 3.0);
    assert_eq!(calls.load(Ordering::Relaxed), 0);
}

#[test]
fn retained_mapping_is_invalidated_by_rules_supports_and_failed_mappers() {
    struct Failing;
    impl FixedRule<f64> for Failing {
        fn algorithm_id(&self) -> scioxide_core::AlgorithmId {
            scioxide_core::algorithm_id!("test", "Failing", 1)
        }
        fn node_count(&self) -> usize {
            20
        }
        fn map_nodes_into(
            &self,
            _: Interval<f64>,
            out: &mut [FixedRuleNode<f64>],
        ) -> Result<(), SciOxideError> {
            out[0] = FixedRuleNode::new(0.0, 100.0, None);
            Err(SciOxideError::numerical_instability(
                NumericalIssue::Breakdown,
                "test mapper failed",
            ))
        }
    }
    let gaussian = GaussianRulePlan::gauss_legendre(20).unwrap();
    let kronrod = GaussKronrod::new(GaussKronrodRule::G7K15).unwrap();
    let mut workspace = gaussian.workspace(1).unwrap();
    let mut out = 0.0;
    let unit = Interval::new(0.0, 1.0).unwrap();
    for interval in [unit, Interval::new(1.0, 2.0).unwrap(), unit] {
        for _ in 0..2 {
            let report = gaussian
                .integrate_into(|x: f64| Ok(x), interval, &mut out, &mut workspace, native())
                .unwrap();
            assert!(report.is_success());
            assert!((out - interval.midpoint().unwrap()).abs() < 1e-14);
        }
    }
    assert!(
        Failing
            .integrate_into(Ok, unit, &mut out, &mut workspace, native())
            .is_err()
    );
    for _ in 0..2 {
        let report = gaussian
            .integrate_into(|x: f64| Ok(x), unit, &mut out, &mut workspace, native())
            .unwrap();
        assert!(report.is_success());
        assert!((out - 0.5).abs() < 1e-14);
        let report = kronrod
            .integrate_into(|x: f64| Ok(x * x), unit, &mut out, &mut workspace, native())
            .unwrap();
        assert!(report.is_success());
        assert!((out - 1.0 / 3.0).abs() < 1e-14);
    }
}

#[test]
fn fused_scalar_sampling_reconciles_failures_and_recovers_overflow_without_replay() {
    let gaussian = GaussianRulePlan::gauss_legendre(20).unwrap();
    let mut workspace = gaussian.workspace(1).unwrap();
    let mut out = 9.0;
    let calls = AtomicUsize::new(0);
    let failed = gaussian.integrate_into(
        |_: f64| {
            let n = calls.fetch_add(1, Ordering::Relaxed);
            if n == 7 {
                Err(SciOxideError::numerical_instability(
                    NumericalIssue::Breakdown,
                    "test callback failed",
                ))
            } else {
                Ok(1.0)
            }
        },
        Interval::new(0.0, 1.0).unwrap(),
        &mut out,
        &mut workspace,
        native(),
    );
    assert!(failed.is_err());
    assert_eq!(out, 9.0);
    assert_eq!(calls.load(Ordering::Relaxed), 8);
    let failed_snapshot = workspace.last_attempt().unwrap();
    assert_eq!(
        workspace
            .last_attempt()
            .unwrap()
            .counts()
            .own()
            .provider(scioxide_core::EvaluationRole::Function),
        8
    );
    calls.store(0, Ordering::Relaxed);
    let report = gaussian
        .integrate_into(
            |x: f64| {
                calls.fetch_add(1, Ordering::Relaxed);
                Ok(if x < 0.0 { -f64::MAX } else { f64::MAX })
            },
            Interval::new(-100.0, 100.0).unwrap(),
            &mut out,
            &mut workspace,
            native(),
        )
        .unwrap();
    assert_eq!(calls.load(Ordering::Relaxed), 20);
    assert_eq!(report.evaluations(), 20);
    assert!(out.abs() < 1e-14);
    assert_eq!(failed_snapshot.outcome(), AttemptOutcome::Failed);
    assert_eq!(
        failed_snapshot
            .counts()
            .own()
            .provider(scioxide_core::EvaluationRole::Function),
        8
    );
    assert!(failed_snapshot.progress().is_none());
}

#[test]
fn retained_mapping_respects_signed_zero_endpoints() {
    fn check<S: RealScalar + Default>() {
        let plan = GaussianRulePlan::with_options(
            OrthogonalFamily::Legendre,
            3,
            GaussianRuleVariant::RadauLeft,
            GaussianRuleOptions::default(),
        )
        .unwrap();
        let mut workspace = plan.workspace(1).unwrap();
        let function = |x: S| {
            Ok(if x == S::ZERO && x.is_sign_negative() {
                S::ONE
            } else {
                S::ZERO
            })
        };
        for lower in [S::ZERO, -S::ZERO, S::ZERO] {
            let interval = Interval::new(lower, S::ONE).unwrap();
            let expected = plan.integrate(&function, interval).unwrap().0;
            let mut actual = S::ZERO;
            assert!(
                plan.integrate_into(&function, interval, &mut actual, &mut workspace, native())
                    .unwrap()
                    .is_success()
            );
            assert_eq!(actual, expected);
        }
    }
    check::<f32>();
    check::<f64>();
}

#[test]
fn every_builtin_has_bit_identical_owned_borrowed_and_erased_contractions() {
    fn check_value<S, R, V>(rule: &R, function: impl Fn(S) -> Result<V, SciOxideError> + Sync)
    where
        S: RealScalar + Default,
        R: FixedRule<S>,
        V: AffineValueMut<Scale = S> + Copy + Default,
    {
        let interval = Interval::new(-S::ONE, S::ONE).unwrap();
        let mut expected = V::default();
        let mut actual = V::default();
        let mut owned = rule.workspace(expected.dimension().unwrap()).unwrap();
        let mut borrowed = rule.workspace(expected.dimension().unwrap()).unwrap();
        let mut erased = rule.workspace(expected.dimension().unwrap()).unwrap();
        for _ in 0..2 {
            let report = R::integrate_into(
                rule,
                &function,
                interval,
                &mut expected,
                &mut owned,
                native(),
            )
            .unwrap();
            let borrowed_report = <&R as FixedRule<S>>::integrate_into(
                &rule,
                &function,
                interval,
                &mut actual,
                &mut borrowed,
                native(),
            )
            .unwrap();
            assert_eq!(borrowed_report, report);
            for c in 0..expected.dimension().unwrap() {
                assert_eq!(
                    actual.component(c).unwrap().into_f64().to_bits(),
                    expected.component(c).unwrap().into_f64().to_bits()
                );
            }
            let object: &dyn FixedRule<S> = rule;
            let erased_report = <&dyn FixedRule<S> as FixedRule<S>>::integrate_into(
                &object,
                &function,
                interval,
                &mut actual,
                &mut erased,
                native(),
            )
            .unwrap();
            assert_eq!(erased_report, report);
            for c in 0..expected.dimension().unwrap() {
                let bits = |v: S| v.into_f64().to_bits();
                assert_eq!(
                    bits(actual.component(c).unwrap()),
                    bits(expected.component(c).unwrap())
                );
                let evidence = |ws: &FixedRuleWorkspace<S>| {
                    ws.component_errors().map(|errors| {
                        let error = errors[c];
                        [
                            error.discretization().map(|e| bits(e.magnitude())),
                            error.roundoff().map(bits),
                            error.total().map(bits),
                        ]
                    })
                };
                assert_eq!(evidence(&owned), evidence(&borrowed));
                assert_eq!(evidence(&owned), evidence(&erased));
            }
        }
    }
    fn check<S, R>(rule: R)
    where
        S: RealScalar + Default,
        R: FixedRule<S>,
    {
        check_value(&rule, |x: S| Ok(x.exp()));
        check_value(&rule, |x: S| Ok(Complex::new(x.exp(), x)));
        check_value(&rule, |x: S| Ok([x.exp(), x, x * x]));
    }
    fn precision<S: RealScalar + NumericScalar<Real = S> + Default>() {
        for pair in [
            GaussKronrodRule::G7K15,
            GaussKronrodRule::G10K21,
            GaussKronrodRule::G15K31,
            GaussKronrodRule::G20K41,
            GaussKronrodRule::G25K51,
        ] {
            check(GaussKronrod::<S>::new(pair).unwrap());
        }
        for variant in [
            GaussianRuleVariant::Gauss,
            GaussianRuleVariant::RadauLeft,
            GaussianRuleVariant::RadauRight,
            GaussianRuleVariant::Lobatto,
        ] {
            check(
                GaussianRulePlan::<S>::with_options(
                    OrthogonalFamily::Legendre,
                    8,
                    variant,
                    GaussianRuleOptions::default(),
                )
                .unwrap(),
            );
        }
        for family in [
            OrthogonalFamily::Jacobi {
                alpha: S::ZERO,
                beta: S::ZERO,
            },
            OrthogonalFamily::Gegenbauer { lambda: S::HALF },
        ] {
            check(GaussianRulePlan::<S>::new(family, 8).unwrap());
        }
        check(ClenshawCurtis::<S>::new(16).unwrap());
        for endpoints in [NewtonCotesEndpoints::Open, NewtonCotesEndpoints::Closed] {
            check(NewtonCotes::<S>::new(3, endpoints, Budget::try_from(4).unwrap()).unwrap());
        }
    }
    precision::<f32>();
    precision::<f64>();
}

#[test]
fn gaussian_unweighted_rejections_forward_before_domain_shortcuts() {
    fn check<S: RealScalar + NumericScalar<Real = S>>() {
        for family in [
            OrthogonalFamily::HermiteH,
            OrthogonalFamily::HermiteHe,
            OrthogonalFamily::Laguerre,
            OrthogonalFamily::GeneralizedLaguerre { alpha: S::HALF },
            OrthogonalFamily::ChebyshevT,
            OrthogonalFamily::ChebyshevU,
            OrthogonalFamily::Jacobi {
                alpha: S::ONE,
                beta: S::ZERO,
            },
            OrthogonalFamily::Gegenbauer { lambda: S::ONE },
        ] {
            let rule = GaussianRulePlan::new(family, 4).unwrap();
            let mut workspace = rule.workspace(1).unwrap();
            let mut out = S::TWO;
            let calls = AtomicUsize::new(0);
            let function = |_: S| {
                calls.fetch_add(1, Ordering::Relaxed);
                Ok(S::ONE)
            };
            for upper in [S::ZERO, S::ONE] {
                let interval = Interval::new(S::ZERO, upper).unwrap();
                let owned = rule
                    .integrate_into(&function, interval, &mut out, &mut workspace, native())
                    .unwrap_err();
                let borrowed = <&GaussianRulePlan<S> as FixedRule<S>>::integrate_into(
                    &&rule,
                    &function,
                    interval,
                    &mut out,
                    &mut workspace,
                    native(),
                )
                .unwrap_err();
                let erased: &dyn FixedRule<S> = &rule;
                let erased = <&dyn FixedRule<S> as FixedRule<S>>::integrate_into(
                    &erased,
                    &function,
                    interval,
                    &mut out,
                    &mut workspace,
                    native(),
                )
                .unwrap_err();
                assert_eq!(owned, borrowed);
                assert_eq!(owned, erased);
                assert_eq!(out, S::TWO);
                assert_eq!(
                    workspace.last_attempt().unwrap().outcome(),
                    AttemptOutcome::PreflightRejected
                );
            }
            let mut workspace = rule.workspace(0).unwrap();
            let mut out = [S::ZERO; 0];
            let interval = Interval::new(S::ZERO, S::ONE).unwrap();
            let function = |_: S| {
                calls.fetch_add(1, Ordering::Relaxed);
                Ok([S::ZERO; 0])
            };
            let owned = rule
                .integrate_into(&function, interval, &mut out, &mut workspace, native())
                .unwrap_err();
            let borrowed = <&GaussianRulePlan<S> as FixedRule<S>>::integrate_into(
                &&rule,
                &function,
                interval,
                &mut out,
                &mut workspace,
                native(),
            )
            .unwrap_err();
            assert_eq!(owned, borrowed);
            assert_eq!(calls.load(Ordering::Relaxed), 0);
        }
    }
    check::<f32>();
    check::<f64>();
}
