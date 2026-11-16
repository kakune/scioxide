//! Scoped accounting for shared Bessel/Faddeeva kernels used by radial plans.
#![allow(clippy::unwrap_used)]
#[path = "support/publication.rs"]
mod publication;
use scioxide_core::{
    Budget, BudgetKind, CancellationToken, Complex, ExecutionContext, ProblemToken, SolveBudgets,
    SolveIdentity, SolveState,
};
use scioxide_special::*;

fn identity() -> SolveIdentity {
    SolveIdentity::new(
        ProblemToken::new(11),
        scioxide_core::algorithm_id!("special", "context", 1),
        0,
        0,
    )
}

#[test]
fn real_elliptic_kernels_preserve_scoped_accounting_with_reused_storage() {
    let options = EllipticOptions::default();
    for scale in [1.0_f64, 2.0_f64.powi(-1000), 2.0_f64.powi(1000)] {
        let expected =
            carlson_rj_with_options(scale, 2.0 * scale, 3.0 * scale, 0.25 * scale, options)
                .unwrap();
        let mut state = SolveState::new();
        let mut budget = SolveBudgets::unbounded();
        let mut context = state
            .begin_solve(ExecutionContext::sequential(), identity(), &mut budget)
            .unwrap();
        let actual = carlson_rj_scoped(
            scale,
            2.0 * scale,
            3.0 * scale,
            0.25 * scale,
            options,
            &mut context,
        )
        .unwrap();
        assert_eq!(actual, expected);
        assert_eq!(
            context.counts().nested().iterations(),
            actual.counts().iterations()
        );
    }
    {
        elliptic_f(0.9_f64, 0.99).unwrap();
        elliptic_e_inc(0.9_f64, 0.99).unwrap();
        legendre_p(50, 0.75_f64).unwrap();
        for m in [-4.0_f64, 0.5, 2.0] {
            let expected = jacobi_sn_with_options(2.5_f64, m, options).unwrap();
            let actual =
                jacobi_sn_scoped(2.5_f64, m, options, ExecutionContext::sequential()).unwrap();
            assert_eq!(actual, expected);
        }
    };
}
#[test]
fn context_and_scalar_reports_match_with_reused_storage() {
    let z = Complex::new(2.0_f64, 0.5);
    let ordinary =
        bessel_j_with_options(Complex::new(0.5, 0.0), z, BesselOptions::default()).unwrap();
    let mut state = SolveState::new();
    let mut budget = SolveBudgets::unbounded();
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), identity(), &mut budget)
        .unwrap();
    let result = bessel_j_scoped(
        Complex::new(0.5, 0.0),
        z,
        BesselOptions::default(),
        &mut context,
    )
    .unwrap();
    assert_eq!(result, ordinary);
    assert_eq!(context.counts().own().iterations(), 0);
    assert_eq!(
        context.counts().nested().iterations(),
        result.counts().iterations()
    );
    let root = bessel_j_scoped(
        Complex::new(0.5, 0.0),
        z,
        BesselOptions::default(),
        ExecutionContext::sequential(),
    )
    .unwrap();
    assert_eq!(root, ordinary);
    let before = context.counts().nested().iterations();
    let result =
        spherical_jn_scoped(3, z, SphericalBesselOptions::default(), &mut context).unwrap();
    assert_eq!(
        result,
        spherical_jn_with_options(3, z, SphericalBesselOptions::default()).unwrap()
    );
    assert_eq!(
        context.counts().nested().iterations() - before,
        result.counts().iterations()
    );
}
#[test]
fn option_limits_cover_bessel_and_dawson_without_charging_outer_iterations() {
    for limit in [1, 2, 8, 32] {
        let budget = Budget::try_from(limit).unwrap();
        // The parent limit is scope-local: it neither bounds nor absorbs child work.
        let mut state = SolveState::new();
        let mut parent_budget =
            SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::ONE);
        let mut context = state
            .begin_solve(
                ExecutionContext::sequential(),
                identity(),
                &mut parent_budget,
            )
            .unwrap();
        let result = bessel_k_scoped(
            Complex::new(2.5_f64, 0.0),
            Complex::new(0.5, 0.25),
            BesselOptions::default().with_iteration_budget(budget),
            &mut context,
        );
        let counts = publication::budget_counts(result);
        assert!(counts.iterations() <= limit);
        assert_eq!(context.counts().own().iterations(), 0);
        assert_eq!(context.counts().nested().iterations(), counts.iterations());
        let mut state = SolveState::new();
        let mut context = state
            .begin_solve(
                ExecutionContext::sequential(),
                identity(),
                &mut parent_budget,
            )
            .unwrap();
        // A real Faddeeva argument charges the fixed-cost Dawson kernel.
        let result = faddeeva_scoped(
            Complex::new(2.0_f64, 0.0),
            ProbabilityOptions::default().with_iteration_budget(budget),
            &mut context,
        )
        .unwrap();
        assert!(result.counts().iterations() <= limit);
        assert_eq!(context.counts().own().iterations(), 0);
        assert_eq!(
            context.counts().nested().iterations(),
            result.counts().iterations()
        );
    }
}
#[test]
fn cancellation_is_not_relabelled_as_iteration_exhaustion() {
    let token = CancellationToken::new();
    token.cancel();
    let mut state = SolveState::new();
    let mut budget = SolveBudgets::unbounded();
    let execution = ExecutionContext::sequential().with_cancellation(&token);
    let mut context = state
        .begin_solve(execution, identity(), &mut budget)
        .unwrap();
    {
        let outcome = &(bessel_j_scoped(
            Complex::new(0.0_f64, 0.0),
            Complex::new(2.0, 0.25),
            BesselOptions::default(),
            &mut context,
        ));
        let err = outcome.as_ref().expect_err("expected error");
        assert!(matches!(
            err.view(),
            scioxide_core::SciOxideErrorView::ExecutionStopped { .. }
        ));
    }
    {
        let outcome = &(faddeeva_scoped(
            Complex::new(1.0_f64, 0.0),
            ProbabilityOptions::default(),
            &mut context,
        ));
        let err = outcome.as_ref().expect_err("expected error");
        assert!(matches!(
            err.view(),
            scioxide_core::SciOxideErrorView::ExecutionStopped { .. }
        ));
    }
    assert_eq!(context.counts().total().unwrap().iterations(), 0);
}

#[test]
fn faddeeva_series_and_reflection_preserve_scope_limits() {
    for z in [
        Complex::new(1e-5_f64, 0.5),
        Complex::new(3.0, 2.0),
        Complex::new(3.0, -2.0),
        Complex::new(1.0, -30.0),
    ] {
        let complete = faddeeva_with_options(z, ProbabilityOptions::default()).unwrap();
        for limit in [1, complete.counts().iterations()] {
            let options = ProbabilityOptions::new(Budget::try_from(limit).unwrap());
            let expected = faddeeva_with_options(z, options).unwrap();
            let mut state = SolveState::new();
            let mut budgets = SolveBudgets::unbounded();
            let mut context = state
                .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
                .unwrap();
            let actual = faddeeva_scoped(z, options, &mut context).unwrap();
            assert_eq!(actual, expected);
            assert_eq!(context.counts().own().iterations(), 0);
            assert_eq!(
                context.counts().nested().iterations(),
                actual.counts().iterations()
            );
        }
        let token = CancellationToken::new();
        token.cancel();
        let error = faddeeva_scoped(
            z,
            ProbabilityOptions::default(),
            ExecutionContext::sequential().with_cancellation(&token),
        )
        .unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::ExecutionStopped { .. }
        ));
    }
}
#[test]
fn overflowing_bessel_retains_binary_scale() {
    let r = bessel_i_with_options(
        Complex::new(0.5_f64, 0.0),
        Complex::new(1000.0, 0.25),
        BesselOptions::default(),
    )
    .unwrap();
    assert_eq!(r.scalar_representation(), ScalarRepresentation::Overflow);
    assert!(r.value().is_none());
    assert!(r.scaled().is_some() && r.logarithmic().is_some());
    let s = r.scaled().unwrap().value();
    assert!(
        (1.0..2.0).contains(
            &s.real()
                .significand()
                .abs()
                .max(s.imaginary().significand().abs())
        )
    );
    assert!(s.try_projective().unwrap().exponent() > 1000);
}
