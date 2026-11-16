//! Shared Airy derivatives, actual admitted work, and full-range real seeds.
#![allow(clippy::unwrap_used)]
#[path = "support/fixture.rs"]
mod fixture;
#[path = "support/publication.rs"]
mod publication;
use scioxide_core::{Budget, BudgetKind, ExecutionContext, RealScalar, SolveBudgets, SolveState};
use scioxide_special::*;

#[test]
fn binary64_overflow_edge_retains_extended_representations() {
    // Independently rounded from the 160/240-digit Bi range fixture.
    let last_finite = f64::from_bits(0x405a_1bea_c387_f62c);
    let first_overflow = f64::from_bits(0x405a_1bea_c387_f62d);
    let finite = airy_bi_with_options(last_finite, AiryOptions::default()).unwrap();
    assert!(finite.value().is_some());
    assert!(finite.derivative().is_none());
    let overflow = airy_bi_with_options(first_overflow, AiryOptions::default()).unwrap();
    assert_eq!(
        overflow.scalar_representation(),
        ScalarRepresentation::Overflow
    );
    assert!(overflow.value().is_none());
    assert!(overflow.scaled().is_some());
    assert!(overflow.logarithmic().is_some());
    {
        let outcome = &(airy_bi(first_overflow));
        let err = outcome.as_ref().expect_err("expected error");
        assert!(matches!(
            err.view(),
            scioxide_core::SciOxideErrorView::NumericalInstability {
                issue: scioxide_core::NumericalIssue::Overflow,
                ..
            }
        ));
    }
}

fn references<S: RealScalar>(width: &str) {
    let mut checker = fixture::Checker::new("airy", width);
    for row in include_str!("data/airy_real.tsv")
        .lines()
        .filter(|l| !l.starts_with('#'))
    {
        let f: Vec<_> = row.split_whitespace().collect();
        if f[0] != width {
            continue;
        }
        let x = S::try_from_f64_lossy(f[2].parse().unwrap()).unwrap();
        let options = AiryOptions::default();
        let r = match f[1] {
            "ai" => airy_ai_with_options(x, options),
            "bi" => airy_bi_with_options(x, options),
            "ai_scaled" => airy_ai_scaled_with_options(x, options),
            _ => airy_bi_scaled_with_options(x, options),
        }
        .unwrap();
        assert!(r.termination().is_success(), "{row}: {r:?}");
        let v = r.scaled().unwrap().value();
        let expected = f[3].parse::<f64>().unwrap();
        let value = scioxide_core::checked_scalbn(
            v.try_projective()
                .unwrap()
                .significand()
                .try_to_f64_exact()
                .unwrap(),
            (v.try_projective().unwrap().exponent() - f[4].parse::<i64>().unwrap()) as i32,
        )
        .unwrap();
        let scaled_error = r.scaled().unwrap().absolute_error().into_f64();
        let exponent = v.try_projective().unwrap().exponent() - f[4].parse::<i64>().unwrap();
        let error = scioxide_core::checked_scalbn(scaled_error, exponent as i32).unwrap();
        checker.sample(
            "value",
            (value - expected).abs(),
            expected.abs(),
            Some(error),
            row,
        );
        let expected =
            scioxide_core::checked_scalbn(f[5].parse::<f64>().unwrap(), f[6].parse().unwrap())
                .ok()
                .and_then(|v| S::try_from_f64_lossy(v).ok())
                .filter(|v| *v != S::ZERO);
        if let Some(expected) = expected {
            let value = *r.derivative().unwrap().value();
            checker.scalar(
                "derivative",
                value,
                expected,
                Some(r.derivative().unwrap().absolute_error()),
                row,
            );
        } else {
            assert!(r.derivative().is_none(), "{row}: {r:?}");
        }
    }
}
#[test]
fn independent_real_f64() {
    references::<f64>("64");
}
#[test]
fn independent_real_f32() {
    references::<f32>("32");
}
#[test]
fn scoped_budgets_and_derivative_rescaling() {
    let identity = scioxide_core::SolveIdentity::new(
        scioxide_core::ProblemToken::new(502),
        scioxide_core::algorithm_id!("Airy", "test", 1),
        0,
        0,
    );
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::ONE);
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
        .unwrap();
    // The parent limit is scope-local; the options budget bounds the child epoch.
    let r = airy_ai_scoped(
        5.0_f64,
        AiryOptions::default().with_iteration_budget(Budget::ONE),
        &mut context,
    )
    .unwrap_err();
    let _ = publication::exhausted_counts(&r);
    assert_eq!(context.counts().own().iterations(), 0);
    assert_eq!(context.counts().nested().iterations(), 1);
    assert!(
        airy_ai_scoped(5.0_f64, AiryOptions::default(), &mut context)
            .unwrap()
            .termination()
            .is_success()
    );
    assert_eq!(
        airy_ai_with_options(1.0_f64, AiryOptions::default())
            .unwrap()
            .termination(),
        SpecialFunctionTermination::Converged
    );
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
        .unwrap();
    let r = airy_bi_scoped(5.0_f64, AiryOptions::default(), &mut context).unwrap();
    assert_eq!(context.counts().own().iterations(), 0);
    assert_eq!(
        context.counts().nested().iterations(),
        r.counts().iterations()
    );
    assert_eq!(
        r,
        airy_bi_scoped(
            5.0_f64,
            AiryOptions::default(),
            ExecutionContext::sequential()
        )
        .unwrap()
    );
    let scaled = airy_bi_scaled_with_options(5.0_f64, AiryOptions::default()).unwrap();
    let factor = (-2.0 * 5.0_f64.powf(1.5) / 3.0).exp();
    assert!(
        (scaled.derivative().unwrap().value() - r.derivative().unwrap().value() * factor).abs()
            < 1e-12
    );
}
