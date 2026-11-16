//! Internal utility helpers shared across all `scioxide-special` modules.
//!
//! All items are `pub(crate)` — they are not part of the public API.

use scioxide_core::{
    Budget, BudgetKind, ProblemToken, RealScalar, SciOxideError, SolveBudgets, SolveContext,
    SolveIdentity, SolveScope, SolveState, checked_midpoint,
};

/// Identity of every scoped special-function epoch.
const SCOPED: SolveIdentity = SolveIdentity::new(
    ProblemToken::new(0),
    scioxide_core::algorithm_id!("special", "scoped", 1),
    0,
    0,
);

/// Run `evaluate` in a fresh epoch of `scope` whose local iteration allowance is `budget`.
///
/// An execution context starts a root epoch and a solve context a child epoch.
/// Ancestors receive the work as nested counts; iteration allowances are
/// scope-local, so `budget` bounds the evaluation while cancellation and
/// deadlines still propagate. No allocation occurs.
pub(crate) fn scoped<'scope, 'cap, T>(
    scope: impl Into<SolveScope<'scope, 'cap>>,
    budget: Budget,
    evaluate: impl FnOnce(&mut SolveContext<'_, 'cap>) -> Result<T, SciOxideError>,
) -> Result<T, SciOxideError> {
    scope.into().run(
        &mut SolveState::new(),
        SCOPED,
        SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, budget),
        evaluate,
    )
}
/// Test whether the complete scalar state is exactly zero.
///
/// Exact-value shortcuts go through this helper so call sites do not duplicate
/// identity comparisons.
pub(crate) fn is_exact_zero<S: RealScalar>(value: S) -> bool {
    value.is_zero()
}

/// Test exact equality without discarding non-primal scalar state.
pub(crate) fn is_exact<S: RealScalar>(left: S, right: S) -> bool {
    is_exact_zero(left - right)
}

/// Return a representable point strictly inside an ordered finite bracket.
///
/// `None` means the endpoints are adjacent and no scalar can make further
/// bracket progress. The midpoint calculation is overflow-safe and each
/// endpoint equality is repaired by one directed representable step.
pub(crate) fn strict_bracket_candidate<S: RealScalar>(
    lower: S,
    upper: S,
) -> Result<Option<S>, SciOxideError> {
    let midpoint = checked_midpoint(lower, upper)?;
    let candidate = if midpoint <= lower {
        lower.next_after(upper)
    } else if midpoint >= upper {
        upper.next_after(lower)
    } else {
        midpoint
    };
    Ok((candidate > lower && candidate < upper).then_some(candidate))
}

/// Replace an underflow-sized continued-fraction state while preserving its
/// sign, as required by modified Lentz updates.
pub(crate) fn lentz_floor<S: RealScalar>(value: S, tiny: S) -> S {
    if value.abs() >= tiny {
        value
    } else {
        tiny.copysign(value)
    }
}

/// Evaluate `sin(pi * x)` after reducing `x` around its nearest integer.
///
/// Forming `pi * x` before range reduction loses all useful phase information
/// close to large integer arguments.  Reducing first also makes exact integer
/// and half-integer values exact in the result.
pub(crate) fn sin_pi<S: RealScalar>(x: S) -> S {
    let nearest = x.round();
    let reduced = x - nearest;
    let two = S::TWO;
    let parity = nearest - two * (nearest / two).floor();
    let sign = if is_exact_zero(parity) {
        S::ONE
    } else {
        -S::ONE
    };

    if is_exact_zero(reduced) {
        return S::ZERO;
    }
    sign * (S::PI * reduced).sin()
}

/// Evaluate `cos(pi * x)` with the same integer-first reduction as [`sin_pi`].
pub(crate) fn cos_pi<S: RealScalar>(x: S) -> S {
    let nearest = x.round();
    let reduced = x - nearest;
    let two = S::TWO;
    let parity = nearest - two * (nearest / two).floor();
    let sign = if is_exact_zero(parity) {
        S::ONE
    } else {
        -S::ONE
    };

    if is_exact(reduced.abs(), S::HALF) {
        return S::ZERO;
    }
    sign * (S::PI * reduced).cos()
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::support::{check_finite, check_positive, checked_probability, require_finite_param};

    #[test]
    fn test_pi_trigonometry_reduces_before_multiplication() {
        assert_eq!(sin_pi(10_000_000.0_f64), 0.0);
        assert_eq!(cos_pi(10_000_000.5_f64), 0.0);
        let offset = 1e-10_f64;
        let value = sin_pi(-3.0 + offset);
        assert!((value + (std::f64::consts::PI * offset).sin()).abs() < 1e-15);
    }

    #[test]
    fn test_checked_probability_rejects_material_overshoot() {
        assert_eq!(checked_probability(-f64::EPSILON).unwrap(), 0.0);
        {
            let outcome = &(checked_probability(-1e-6_f64));
            let err = outcome.as_ref().expect_err("expected error");
            assert!(matches!(
                err.view(),
                scioxide_core::SciOxideErrorView::NumericalInstability { .. }
            ));
        }
    }

    #[test]
    fn test_lentz_floor_preserves_sign() {
        assert_eq!(lentz_floor(-0.0_f64, 1e-200), -1e-200);
        assert_eq!(lentz_floor(0.0_f64, 1e-200), 1e-200);
        assert_eq!(lentz_floor(-0.0_f32, 1e-20), -1e-20);
        assert_eq!(lentz_floor(0.0_f32, 1e-20), 1e-20);
        assert_eq!(lentz_floor(1e-300_f64, 1e-200), 1e-200);
        assert_eq!(lentz_floor(-1e-300_f64, 1e-200), -1e-200);
        assert_eq!(lentz_floor(-2e-200_f64, 1e-200), -2e-200);
    }

    #[test]
    fn strict_bracket_candidate_handles_extreme_and_adjacent_brackets() {
        assert_eq!(
            strict_bracket_candidate(-f64::MAX, f64::MAX).unwrap(),
            Some(0.0)
        );
        let upper = 1.0_f64.next_after(f64::INFINITY);
        assert_eq!(strict_bracket_candidate(1.0, upper).unwrap(), None);
    }

    #[test]
    fn test_check_finite_ok() {
        assert!(check_finite(1.0_f64).is_ok());
    }

    #[test]
    fn test_check_finite_nan() {
        {
            let outcome = &(check_finite(f64::NAN));
            let err = outcome.as_ref().expect_err("expected error");
            assert!(matches!(
                err.view(),
                scioxide_core::SciOxideErrorView::NonFinite { .. }
            ));
        }
    }

    #[test]
    fn test_check_finite_inf() {
        {
            let outcome = &(check_finite(f64::INFINITY));
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

    // ---- require_finite_param ----

    #[test]
    fn test_require_finite_param_ok() {
        assert!(
            require_finite_param(1.0_f64, "special-function parameter x must be finite").is_ok()
        );
    }

    #[test]
    fn test_require_finite_param_nan() {
        {
            let outcome =
                &(require_finite_param(f64::NAN, "special-function parameter x must be finite"));
            let err = outcome.as_ref().expect_err("expected error");
            assert!(matches!(
                err.view(),
                scioxide_core::SciOxideErrorView::NonFinite {
                    origin: scioxide_core::NonFiniteOrigin::Input,
                    ..
                }
            ));
        }
    }

    #[test]
    fn test_require_finite_param_inf() {
        {
            let outcome = &(require_finite_param(
                f64::INFINITY,
                "special-function parameter x must be finite",
            ));
            let err = outcome.as_ref().expect_err("expected error");
            assert!(matches!(
                err.view(),
                scioxide_core::SciOxideErrorView::NonFinite {
                    origin: scioxide_core::NonFiniteOrigin::Input,
                    ..
                }
            ));
        }
    }

    // ---- check_positive ----

    #[test]
    fn test_check_positive_ok() {
        assert!(
            check_positive(
                1.0_f64,
                scioxide_core::argument!("x"),
                "special-function parameter x must be finite and positive"
            )
            .is_ok()
        );
    }

    #[test]
    fn test_check_positive_zero() {
        {
            let outcome = &(check_positive(
                0.0_f64,
                scioxide_core::argument!("x"),
                "special-function parameter x must be finite",
            ));
            let err = outcome.as_ref().expect_err("expected error");
            assert!(matches!(
                err.view(),
                scioxide_core::SciOxideErrorView::DomainError {
                    reason: scioxide_core::DomainReason::Pole,
                    ..
                }
            ));
        }
    }

    #[test]
    fn test_check_positive_negative() {
        {
            let outcome = &(check_positive(
                -1.0_f64,
                scioxide_core::argument!("x"),
                "special-function parameter x must be finite",
            ));
            let err = outcome.as_ref().expect_err("expected error");
            assert!(matches!(
                err.view(),
                scioxide_core::SciOxideErrorView::DomainError {
                    reason: scioxide_core::DomainReason::OutsideDomain,
                    ..
                }
            ));
        }
    }
}
