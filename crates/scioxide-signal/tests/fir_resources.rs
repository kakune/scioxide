//! Prepared allocation, transactional output, executor and solve-budget tests.
#![allow(clippy::unwrap_used)]
#[path = "support/errors.rs"]
mod errors;
use errors::error_kind;

use scioxide_core::{
    Budget, BudgetKind, CancellationToken, ExecutionContext, ProblemToken, SciOxideErrorKind,
    SolveBudgets, SolveIdentity, SolveState,
};
use scioxide_signal::*;

fn plan() -> SavitzkyGolayPlan<f64> {
    SavitzkyGolayPlan::new(5, 2, SavitzkyGolayOptions::new().with_derivative(1)).unwrap()
}

// Preparation keeps O(w*p) fit and evaluation tables instead of a w×w
// endpoint table (8*w² = 128 MB of f64 at w = 4001).

#[test]
fn boundary_values_and_transactional_failure() {
    let p = plan();
    let input = [1., 3., 8., 2., 4., 9., 1.];
    let mut out = [71.; 9];
    let mut scratch = p.workspace(SignalLayout::Contiguous(7)).unwrap();
    for b in [
        SignalBoundary::Zero,
        SignalBoundary::Periodic,
        SignalBoundary::Reflect,
    ] {
        p.filter_into(
            &input,
            SavitzkyGolayBoundary::Extend(b),
            &mut out,
            &mut scratch,
            ExecutionContext::sequential(),
        )
        .unwrap();
        for (i, &actual) in out.iter().enumerate().take(input.len()) {
            let expected = p
                .coefficients()
                .iter()
                .enumerate()
                .map(|(j, &h)| {
                    let k = i as i64 + j as i64 - 2;
                    let v = match b {
                        SignalBoundary::Zero => {
                            if (0..7).contains(&k) {
                                input[k as usize]
                            } else {
                                0.
                            }
                        }
                        SignalBoundary::Periodic => input[k.rem_euclid(7) as usize],
                        SignalBoundary::Reflect => {
                            let q = k.rem_euclid(12);
                            input[if q < 7 { q } else { 12 - q } as usize]
                        }
                    };
                    h * v
                })
                .sum::<f64>();
            assert!((actual - expected).abs() < 1e-12);
        }
        assert_eq!(&out[7..], &[71., 71.]);
    }
    let before = out;
    let mut bad = input;
    bad[6] = f64::NAN;
    assert_eq!(
        error_kind(p.filter_into(
            &bad,
            SavitzkyGolayBoundary::Polynomial,
            &mut out,
            &mut scratch,
            ExecutionContext::sequential()
        )),
        SciOxideErrorKind::NonFinite
    );
    assert_eq!(out, before);
    assert_eq!(
        error_kind(p.filter_into(
            &input,
            SavitzkyGolayBoundary::Polynomial,
            &mut out,
            &mut p.workspace(SignalLayout::Contiguous(6)).unwrap(),
            ExecutionContext::sequential()
        )),
        SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(out, before);
    let cancellation = CancellationToken::new();
    cancellation.cancel();
    assert_eq!(
        error_kind(p.filter_into(
            &input,
            SavitzkyGolayBoundary::Polynomial,
            &mut out,
            &mut scratch,
            ExecutionContext::sequential().with_cancellation(&cancellation)
        )),
        SciOxideErrorKind::ExecutionStopped
    );
    assert_eq!(out, before);
    let high = SavitzkyGolayPlan::new(
        3,
        1,
        SavitzkyGolayOptions::new()
            .with_derivative(1)
            .with_spacing(0.01),
    )
    .unwrap();
    assert_eq!(
        error_kind(high.filter_into(
            &[-f64::MAX, 0., f64::MAX],
            SavitzkyGolayBoundary::Polynomial,
            &mut out,
            &mut scratch,
            ExecutionContext::sequential()
        )),
        SciOxideErrorKind::NumericalInstability
    );
    assert_eq!(out, before);
}
fn identity() -> SolveIdentity {
    SolveIdentity::new(
        ProblemToken::new(1),
        scioxide_core::algorithm_id!("signal", "fir-design", 1),
        1,
        0,
    )
}
#[test]
fn default_designs_match_explicit_native_context() {
    let bands = [
        FirBand::new(0.0, 0.3, 1.0, 1.0),
        FirBand::new(0.4, 1.0, 0.0, 0.0),
    ];
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut context = state
        .begin_solve(
            ExecutionContext::sequential_native(),
            identity(),
            &mut budgets,
        )
        .unwrap();
    let root =
        design_fir_remez_with_options(33, FirSymmetry::Even, &bands, RemezOptions::default())
            .unwrap();
    let child = design_fir_remez_scoped(
        33,
        FirSymmetry::Even,
        &bands,
        RemezOptions::default(),
        &mut context,
    )
    .unwrap();
    assert_eq!(root.coefficients(), child.coefficients());
    assert_eq!(context.counts().nested().iterations(), child.iterations());
    // The owned design is the scoped one in a fresh native root scope.
    let one_shot = design_fir_remez(33, FirSymmetry::Even, &bands).unwrap();
    let native = ExecutionContext::sequential_native();
    let scoped = design_fir_remez_scoped(
        33,
        FirSymmetry::Even,
        &bands,
        RemezOptions::default(),
        native,
    )
    .unwrap();
    let bits = |design: &FirDesign<f64>| -> Vec<u64> {
        design.coefficients().iter().map(|h| h.to_bits()).collect()
    };
    assert_eq!(bits(&one_shot), bits(&scoped));
    let owned = plan();
    let native = SavitzkyGolayPlan::build(
        owned.window_length(),
        owned.polynomial_degree(),
        owned.options(),
        ExecutionContext::sequential_native(),
    )
    .unwrap();
    for (a, b) in owned.coefficients().iter().zip(native.coefficients()) {
        assert!((a - b).abs() <= 1e-12);
    }
}
#[test]
fn shared_iteration_budget() {
    let bands = [FirBand::new(0., 0.3, 1., 1.), FirBand::new(0.4, 1., 0., 0.)];
    let mut state = SolveState::new();
    let mut budgets =
        SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::try_from(1).unwrap());
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
        .unwrap();
    // The exchange runs in a child epoch: its scope-local iterations reach the
    // parent as nested counts and do not consume the parent's allowance.
    let options = RemezOptions::default().with_max_iterations(1);
    let result =
        design_fir_remez_scoped(31, FirSymmetry::Even, &bands, options, &mut context).unwrap();
    assert_eq!(result.termination(), FirTermination::IterationLimit);
    assert_eq!(result.iterations(), 1);
    assert_eq!(context.counts().own().iterations(), 0);
    assert_eq!(context.counts().nested().iterations(), 1);
    let limited = design_fir_remez_with_options(31, FirSymmetry::Even, &bands, options).unwrap();
    assert_eq!(limited.termination(), FirTermination::IterationLimit);
    assert_eq!(limited.coefficients(), result.coefficients());
    assert_eq!(limited.extrema(), result.extrema());
    assert_eq!(limited.weighted_ripple(), result.weighted_ripple());
}

#[test]
fn remez_rejects_stopped_execution() {
    let bands = [FirBand::new(0.0_f64, 0.8, 0.0, 1.0)];
    let cancellation = CancellationToken::new();
    cancellation.cancel();
    let error = design_fir_remez_scoped(
        31,
        FirSymmetry::Even,
        &bands,
        RemezOptions::default(),
        ExecutionContext::sequential().with_cancellation(&cancellation),
    )
    .unwrap_err();
    assert_eq!(
        error.kind(),
        scioxide_core::SciOxideErrorKind::ExecutionStopped
    );

    let cancellation = CancellationToken::new();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut context = state
        .begin_solve(
            ExecutionContext::sequential().with_cancellation(&cancellation),
            identity(),
            &mut budgets,
        )
        .unwrap();
    cancellation.cancel();
    let error = design_fir_remez_scoped(
        31,
        FirSymmetry::Even,
        &bands,
        RemezOptions::default(),
        &mut context,
    )
    .unwrap_err();
    assert_eq!(
        error.kind(),
        scioxide_core::SciOxideErrorKind::ExecutionStopped
    );
    assert_eq!(context.counts().own().iterations(), 0);
    assert_eq!(context.counts().nested().iterations(), 0);
}
