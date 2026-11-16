//! Automatic search reach, domain isolation and admitted callback work.
#![expect(
    clippy::unwrap_used,
    reason = "Tests require successful setup and inspect expected results."
)]

use core::cell::Cell;
use core::ops::Bound::{Excluded, Included, Unbounded};
use scioxide_core::{
    Budget, BudgetKind, CancellationToken, ConvergenceReason, EvaluationRole, ExecutionContext,
    ProblemToken, RealScalar, SciOxideErrorKind, SciOxideErrorView, SolveBudgets, SolveState,
};
use scioxide_rootfind::{
    BracketSearchOptions, BracketVerification, ScalarFn, find_bracket_scoped,
    find_bracket_with_options,
};

fn reach<S: RealScalar>(large: S, tiny: S) {
    for root in [S::TEN + S::TEN, S::TEN.powi(6), large, -large, -tiny] {
        let calls = Cell::new(0);
        let function = ScalarFn::new(|x: S| {
            calls.set(calls.get() + 1);
            Ok(x - root)
        });
        let bracket = find_bracket_with_options(
            &function,
            scioxide_rootfind::BracketSearchInput::Guess(S::ZERO),
            scioxide_rootfind::BracketSearchDirection::Automatic,
            &BracketSearchOptions::default(),
        )
        .unwrap();
        assert!(bracket.interval().contains(root));
        let logarithmic_bound = root.abs().max(S::ONE).log2().into_f64().ceil() as usize + 8;
        assert!(
            calls.get() <= logarithmic_bound,
            "root={root:?}, calls={}",
            calls.get()
        );
        assert_eq!(
            bracket.verify(&function).unwrap(),
            BracketVerification::Verified
        );
    }
    for root in [-S::MAX, S::MAX] {
        // Scale the residual so the callback itself cannot overflow.
        let function = ScalarFn::new(|x: S| Ok(x * S::HALF - root * S::HALF));
        let bracket = find_bracket_with_options(
            &function,
            scioxide_rootfind::BracketSearchInput::Guess(S::ZERO),
            scioxide_rootfind::BracketSearchDirection::Automatic,
            &BracketSearchOptions::default(),
        )
        .unwrap();
        assert!(bracket.interval().contains(root));
        assert!(bracket.lower().is_finite() && bracket.upper().is_finite());
    }
}

fn domains<S: RealScalar>() {
    let options = BracketSearchOptions::default().with_domain(Excluded(S::ZERO), Unbounded);
    assert_eq!(options.domain(), (Excluded(S::ZERO), Unbounded));
    let function = ScalarFn::new(|x: S| {
        assert!(x > S::ZERO && x.is_finite());
        Ok(x.ln() - S::TWO)
    });
    let bracket = find_bracket_with_options(
        &function,
        scioxide_rootfind::BracketSearchInput::Guess(S::ONE),
        scioxide_rootfind::BracketSearchDirection::Automatic,
        &options,
    )
    .unwrap();
    assert!(bracket.interval().contains(S::TWO.exp()));

    // The upper side is blocked immediately, but the lower side has a root.
    let options = BracketSearchOptions::default().with_domain(Unbounded, Included(S::ONE));
    let bracket = find_bracket_with_options(
        &ScalarFn::new(|x: S| Ok(x + S::TEN)),
        scioxide_rootfind::BracketSearchInput::Guess(S::ZERO),
        scioxide_rootfind::BracketSearchDirection::Automatic,
        &options,
    )
    .unwrap();
    assert!(bracket.interval().contains(-S::TEN));

    let options = BracketSearchOptions::default().with_domain(Excluded(-S::ONE), Excluded(S::ONE));
    let bracket = find_bracket_with_options(
        &ScalarFn::new(|x: S| {
            assert!(x > -S::ONE && x < S::ONE);
            Ok(x)
        }),
        scioxide_rootfind::BracketSearchInput::Guess(S::ZERO),
        scioxide_rootfind::BracketSearchDirection::Automatic,
        &options,
    )
    .unwrap();
    assert!(bracket.interval().contains(S::ZERO));

    let calls = Cell::new(0);
    let constant = ScalarFn::new(|x: S| {
        calls.set(calls.get() + 1);
        assert!(x >= S::ZERO && x <= S::ONE);
        Ok(S::ONE)
    });
    let bounded = BracketSearchOptions::default().with_domain(Included(S::ZERO), Included(S::ONE));
    let error = find_bracket_with_options(
        &constant,
        scioxide_rootfind::BracketSearchInput::Guess(S::HALF),
        scioxide_rootfind::BracketSearchDirection::Automatic,
        &bounded,
    )
    .unwrap_err();
    assert!(
        matches!(error.view(), SciOxideErrorView::ConvergenceFailure {
        reason: ConvergenceReason::Stagnation, diagnostics, ..
    } if diagnostics.counts().own().provider(EvaluationRole::Function) == calls.get())
    );
    assert_eq!(calls.get(), 2);

    let singleton = bounded.with_domain(Included(S::ZERO), Included(S::ZERO));
    calls.set(0);
    assert!(
        find_bracket_with_options(
            &constant,
            scioxide_rootfind::BracketSearchInput::Guess(S::ZERO),
            scioxide_rootfind::BracketSearchDirection::Automatic,
            &singleton
        )
        .is_err()
    );
    assert_eq!(calls.get(), 1);
    let bracket = find_bracket_with_options(
        &ScalarFn::new(|x: S| Ok(x)),
        scioxide_rootfind::BracketSearchInput::Guess(S::ZERO),
        scioxide_rootfind::BracketSearchDirection::Automatic,
        &singleton,
    )
    .unwrap();
    assert_eq!(bracket.lower(), S::ZERO);
    assert_eq!(bracket.upper(), S::ZERO);

    calls.set(0);
    let error = find_bracket_with_options(
        &constant,
        scioxide_rootfind::BracketSearchInput::Guess(S::ZERO),
        scioxide_rootfind::BracketSearchDirection::Automatic,
        &options.with_domain(Excluded(S::ZERO), Unbounded),
    )
    .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::InvalidInput);
    assert_eq!(calls.get(), 0);
    for (lower, upper) in [
        (Included(S::ONE), Included(S::ZERO)),
        (Excluded(S::ZERO), Included(S::ZERO)),
        (Excluded(S::MAX), Unbounded),
        (Unbounded, Excluded(-S::MAX)),
        (Excluded(S::ZERO), Excluded(S::MIN_SUBNORMAL)),
    ] {
        assert_eq!(
            find_bracket_with_options(
                &constant,
                scioxide_rootfind::BracketSearchInput::Guess(S::HALF),
                scioxide_rootfind::BracketSearchDirection::Automatic,
                &bounded.with_domain(lower, upper)
            )
            .unwrap_err()
            .kind(),
            SciOxideErrorKind::InvalidInput
        );
    }
    for value in [S::NAN, S::INFINITY, S::NEG_INFINITY] {
        assert_eq!(
            find_bracket_with_options(
                &constant,
                scioxide_rootfind::BracketSearchInput::Guess(S::HALF),
                scioxide_rootfind::BracketSearchDirection::Automatic,
                &bounded.with_domain(Included(value), Unbounded)
            )
            .unwrap_err()
            .kind(),
            SciOxideErrorKind::NonFinite
        );
        assert_eq!(
            find_bracket_with_options(
                &constant,
                scioxide_rootfind::BracketSearchInput::Guess(S::HALF),
                scioxide_rootfind::BracketSearchDirection::Automatic,
                &bounded.with_domain(Unbounded, Excluded(value))
            )
            .unwrap_err()
            .kind(),
            SciOxideErrorKind::NonFinite
        );
    }
}

#[test]
fn double_precision_reach_and_domains() {
    reach(1e300_f64, 1e-300);
    domains::<f64>();
}

#[test]
fn single_precision_reach_and_domains() {
    reach(1e30_f32, 1e-30);
    domains::<f32>();
}

#[test]
fn budgets_cancellation_and_domain_preflight_precede_callbacks() {
    let calls = Cell::new(0);
    let cancellation = CancellationToken::new();
    let options = BracketSearchOptions::<f64>::default()
        .with_budgets(SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, Budget::ONE));
    let function = ScalarFn::new(|x: f64| {
        calls.set(calls.get() + 1);
        Ok(x - 20.0)
    });
    let outside = options.with_domain(Included(1.0), Unbounded);
    assert_eq!(
        find_bracket_with_options(
            &function,
            scioxide_rootfind::BracketSearchInput::Guess(0.0),
            scioxide_rootfind::BracketSearchDirection::Automatic,
            &outside
        )
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::InvalidInput
    );
    assert_eq!(calls.get(), 0);
    let identity = scioxide_core::SolveIdentity::new(
        ProblemToken::new(0),
        scioxide_core::algorithm_id!("rootfind-test", "parent", 1),
        1,
        0,
    );
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut parent = state
        .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
        .unwrap();
    assert_eq!(
        find_bracket_scoped(
            &function,
            scioxide_rootfind::BracketSearchInput::Guess(0.0),
            scioxide_rootfind::BracketSearchDirection::Automatic,
            &options,
            &mut parent
        )
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::ConvergenceFailure
    );
    assert_eq!(calls.get(), 1);
    assert_eq!(
        parent.counts().nested().provider(EvaluationRole::Function),
        1
    );

    let function = ScalarFn::new(|_: f64| {
        calls.set(calls.get() + 1);
        cancellation.cancel();
        // Even an exact endpoint root cannot suppress a requested stop.
        Ok(0.0)
    });
    assert_eq!(
        find_bracket_scoped(
            &function,
            scioxide_rootfind::BracketSearchInput::Guess(0.0),
            scioxide_rootfind::BracketSearchDirection::Automatic,
            &BracketSearchOptions::default(),
            ExecutionContext::sequential().with_cancellation(&cancellation),
        )
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::ExecutionStopped
    );
    assert_eq!(calls.get(), 2);
}

fn monotone_searches<S: RealScalar>() {
    use scioxide_rootfind::{
        BracketExpansion, BracketSearchDirection as Direction, BracketSearchInput as Input,
    };
    // Positive and negative domains, both function directions, both expansion directions.
    for sign in [-S::ONE, S::ONE] {
        let domain = if sign > S::ZERO {
            (Excluded(S::ZERO), Unbounded)
        } else {
            (Unbounded, Excluded(S::ZERO))
        };
        for root in [sign * S::TWO.powi(8), sign / S::TWO.powi(8)] {
            for direction in [Direction::Increasing, Direction::Decreasing] {
                let calls = core::cell::RefCell::new(Vec::new());
                let function = ScalarFn::new(|x: S| {
                    assert!(x * sign > S::ZERO);
                    // A direct additive probe near zero would be unrepresentable.
                    assert!(x.abs() >= S::ONE / S::TWO.powi(10));
                    calls.borrow_mut().push(x);
                    let residual = S::ONE / root - S::ONE / x;
                    Ok(if direction == Direction::Increasing {
                        residual
                    } else {
                        -residual
                    })
                });
                let options = BracketSearchOptions::default()
                    .with_expansion(BracketExpansion::Multiplicative)
                    .with_domain(domain.0, domain.1);
                let bracket =
                    find_bracket_with_options(&function, Input::Guess(sign), direction, &options)
                        .unwrap();
                assert!(bracket.interval().contains(root));
                let calls = calls.borrow();
                assert_eq!(calls[0], sign);
                assert!(calls.windows(2).all(|pair| if root > sign {
                    pair[1] > pair[0]
                } else {
                    pair[1] < pair[0]
                }));
                assert!(calls.len() <= 9);
            }
        }
    }
    // Preserve paired seeds and evaluate each once before expansion.
    let calls = core::cell::RefCell::new(Vec::new());
    let function = ScalarFn::new(|x: S| {
        calls.borrow_mut().push(x);
        Ok(x - S::TEN)
    });
    let bracket = find_bracket_with_options(
        &function,
        Input::Pair(S::ONE, S::TWO),
        Direction::Increasing,
        &BracketSearchOptions::default(),
    )
    .unwrap();
    assert!(bracket.interval().contains(S::TEN));
    assert_eq!(&calls.borrow()[..2], &[S::ONE, S::TWO]);
    assert!(calls.borrow().windows(2).all(|p| p[1] > p[0]));
    let singleton = find_bracket_with_options(
        &ScalarFn::new(|x: S| Ok(x)),
        Input::Pair(S::ZERO, S::ZERO),
        Direction::Increasing,
        &BracketSearchOptions::default(),
    )
    .unwrap();
    assert_eq!(singleton.lower(), S::ZERO);
    assert_eq!(singleton.upper(), S::ZERO);
}

#[test]
fn monotone_and_multiplicative_searches_in_both_precisions() {
    monotone_searches::<f32>();
    monotone_searches::<f64>();
}

#[test]
fn monotone_preflight_domain_clamping_and_counted_exhaustion() {
    use scioxide_rootfind::{
        BracketExpansion, BracketSearchDirection as Direction, BracketSearchInput as Input,
    };
    let calls = Cell::new(0);
    let f = ScalarFn::new(|x: f64| {
        calls.set(calls.get() + 1);
        Ok(x - 10.0)
    });
    let base = BracketSearchOptions::default();
    for (input, options, kind) in [
        (Input::Pair(2.0, 1.0), base, SciOxideErrorKind::InvalidInput),
        (
            Input::Pair(1.0, f64::NAN),
            base,
            SciOxideErrorKind::NonFinite,
        ),
        (
            Input::Guess(1.0),
            base.with_expansion(BracketExpansion::Multiplicative),
            SciOxideErrorKind::InvalidInput,
        ),
        (
            Input::Guess(1.0),
            base.with_expansion_factor(f64::NAN),
            SciOxideErrorKind::NonFinite,
        ),
        (
            Input::Guess(1.0),
            base.with_expansion_factor(1.0),
            SciOxideErrorKind::InvalidInput,
        ),
    ] {
        assert_eq!(
            find_bracket_with_options(&f, input, Direction::Increasing, &options)
                .unwrap_err()
                .kind(),
            kind
        );
        assert_eq!(calls.get(), 0);
    }
    let options = base
        .with_expansion(BracketExpansion::Multiplicative)
        .with_domain(Excluded(0.0), Included(10.0));
    let root =
        find_bracket_with_options(&f, Input::Guess(1.0), Direction::Increasing, &options).unwrap();
    assert_eq!(root.upper(), 10.0);
    calls.set(0);
    let error = find_bracket_with_options(
        &f,
        Input::Guess(1.0),
        Direction::Increasing,
        &options.with_domain(Excluded(0.0), Included(9.0)),
    )
    .unwrap_err();
    assert!(
        matches!(error.view(), SciOxideErrorView::ConvergenceFailure { reason: ConvergenceReason::Stagnation, diagnostics, .. }
        if diagnostics.counts().own().provider(EvaluationRole::Function) == calls.get())
    );
    calls.set(0);
    let error = find_bracket_with_options(
        &f,
        Input::Guess(1.0),
        Direction::Increasing,
        &options.with_max_expansions(Budget::ONE),
    )
    .unwrap_err();
    assert!(
        matches!(error.view(), SciOxideErrorView::ConvergenceFailure { reason: ConvergenceReason::IterationLimit, diagnostics, .. }
        if diagnostics.counts().own().iterations() == 1 && diagnostics.counts().own().provider(EvaluationRole::Function) == 2)
    );
}

#[test]
fn monotone_search_charges_ancestors_and_honors_cancellation_at_an_exact_zero() {
    use scioxide_rootfind::{
        BracketExpansion, BracketSearchDirection as Direction, BracketSearchInput as Input,
    };
    let options = BracketSearchOptions::default()
        .with_expansion(BracketExpansion::Multiplicative)
        .with_domain(Excluded(0.0), Unbounded);
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, Budget::ONE);
    let identity = scioxide_core::SolveIdentity::new(
        ProblemToken::new(0),
        scioxide_core::algorithm_id!("rootfind-test", "parent", 1),
        1,
        0,
    );
    let mut parent = state
        .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
        .unwrap();
    let calls = Cell::new(0);
    let function = ScalarFn::new(|x: f64| {
        calls.set(calls.get() + 1);
        Ok(x - 20.0)
    });
    let error = find_bracket_scoped(
        &function,
        Input::Guess(1.0),
        Direction::Increasing,
        &options,
        &mut parent,
    )
    .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::ConvergenceFailure);
    assert_eq!(calls.get(), 1);
    assert_eq!(parent.counts().own().provider(EvaluationRole::Function), 0);
    assert_eq!(
        parent.counts().nested().provider(EvaluationRole::Function),
        1
    );
    let cancellation = CancellationToken::new();
    let function = ScalarFn::new(|_: f64| {
        cancellation.cancel();
        Ok(0.0)
    });
    assert_eq!(
        find_bracket_scoped(
            &function,
            Input::Guess(1.0),
            Direction::Increasing,
            &options,
            ExecutionContext::sequential().with_cancellation(&cancellation)
        )
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::ExecutionStopped
    );
}

#[test]
fn multiplicative_expansion_saturates_at_finite_and_subnormal_limits() {
    use scioxide_rootfind::{
        BracketExpansion, BracketSearchDirection as Direction, BracketSearchInput as Input,
    };
    for sign in [-1.0_f64, 1.0] {
        let domain = if sign > 0.0 {
            (Excluded(0.0), Unbounded)
        } else {
            (Unbounded, Excluded(0.0))
        };
        let options = BracketSearchOptions::default()
            .with_expansion(BracketExpansion::Multiplicative)
            .with_domain(domain.0, domain.1);
        let maximum = sign * f64::MAX;
        let function = ScalarFn::new(|x: f64| Ok(x * 0.5 - maximum * 0.5));
        let bracket = find_bracket_with_options(
            &function,
            Input::Guess(maximum * 0.75),
            Direction::Increasing,
            &options,
        )
        .unwrap();
        assert!(bracket.interval().contains(maximum));
        let tiny = sign * f64::from_bits(1);
        let function = ScalarFn::new(|x: f64| Ok(x - tiny));
        let bracket = find_bracket_with_options(
            &function,
            Input::Guess(tiny * 2.0),
            Direction::Increasing,
            &options,
        )
        .unwrap();
        assert!(bracket.interval().contains(tiny));
    }
}

#[test]
fn automatic_multiplicative_search_starts_at_the_guess_and_can_change_sides() {
    use scioxide_rootfind::{BracketExpansion, BracketSearchDirection, BracketSearchInput};
    let options = BracketSearchOptions::default()
        .with_expansion(BracketExpansion::Multiplicative)
        .with_domain(Excluded(0.0), Unbounded);
    for root in [0.125, 8.0, 1.0] {
        let calls = core::cell::RefCell::new(Vec::new());
        let f = ScalarFn::new(|x: f64| {
            calls.borrow_mut().push(x);
            Ok(x - root)
        });
        let bracket = find_bracket_with_options(
            &f,
            BracketSearchInput::Guess(1.0),
            BracketSearchDirection::Automatic,
            &options,
        )
        .unwrap();
        assert!(bracket.interval().contains(root));
        assert_eq!(calls.borrow()[0], 1.0);
        if root == 1.0 {
            assert_eq!(calls.borrow().len(), 1);
        }
    }
}
