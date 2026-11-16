//! Positive limits on iterations, evaluations and other solver resources.

use core::fmt;
use core::num::NonZeroUsize;

use crate::error::{InvalidInputReason, SciOxideError};

/// A positive limit on a solver resource, such as iterations or evaluations.
///
/// The option that holds a `Budget` says what it counts. A budget is never
/// zero. Create one from a constant with [`budget!`](crate::budget!), which is
/// checked at compile time, from a run-time `usize` with [`TryFrom`], which
/// returns `InvalidInput` (`OutOfRange`) for zero, or from a [`NonZeroUsize`]
/// with [`From`].
///
/// # Examples
///
/// ```
/// use scioxide_core::{Budget, budget};
///
/// // A constant is checked at compile time, a run-time value on conversion.
/// assert_eq!(budget!(100).get(), 100);
/// let rows = 25;
/// assert_eq!(Budget::try_from(4 * rows)?.get(), 100);
/// assert!(Budget::try_from(0).is_err());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[must_use]
pub struct Budget(NonZeroUsize);

impl fmt::Display for Budget {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.get().fmt(formatter)
    }
}

impl Budget {
    /// The smallest valid budget.
    pub const ONE: Self = Self(NonZeroUsize::MIN);

    /// Returns the limit.
    #[must_use]
    pub const fn get(self) -> usize {
        self.0.get()
    }

    /// Returns the budget increased by `additional`, saturating at `usize::MAX`.
    pub const fn saturating_add(self, additional: usize) -> Self {
        Self(self.0.saturating_add(additional))
    }
}

/// Creates a [`Budget`](crate::Budget) from a positive constant, checked at compile time.
///
/// The argument is any constant `usize` expression, such as `budget!(256)`,
/// `budget!(64 * 1024)` or `budget!(usize::MAX)`. Zero and expressions that are
/// not constant fail to compile; convert a run-time value with
/// `Budget::try_from`.
///
/// # Examples
///
/// ```
/// use scioxide_core::{Budget, budget};
///
/// const MAX_TERMS: Budget = budget!(256);
/// assert_eq!(MAX_TERMS.get(), 256);
/// assert_eq!(budget!(1), Budget::ONE);
/// ```
///
/// Zero does not compile:
///
/// ```compile_fail
/// let _ = scioxide_core::budget!(0);
/// ```
#[macro_export]
macro_rules! budget {
    ($value:expr) => {
        const {
            match $value {
                0_usize => panic!("budget must be greater than zero"),
                value => $crate::Budget::ONE.saturating_add(value - 1),
            }
        }
    };
}

/// Validate a positive count; zero returns `InvalidInput`.
impl TryFrom<usize> for Budget {
    type Error = SciOxideError;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        NonZeroUsize::new(value).map(Self).ok_or_else(|| {
            SciOxideError::invalid_input(
                crate::argument!("value"),
                InvalidInputReason::OutOfRange,
                "budget must be greater than zero",
            )
        })
    }
}

/// Consume an already validated positive count.
impl From<NonZeroUsize> for Budget {
    fn from(value: NonZeroUsize) -> Self {
        Self(value)
    }
}

impl From<Budget> for NonZeroUsize {
    fn from(value: Budget) -> Self {
        value.0
    }
}

impl From<Budget> for usize {
    fn from(value: Budget) -> Self {
        value.get()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positive_budget_round_trips() {
        for count in [1, 64, usize::MAX] {
            let budget = Budget::try_from(count).unwrap();
            let nonzero = NonZeroUsize::from(budget);
            assert_eq!(budget.get(), count);
            assert_eq!(usize::from(budget), count);
            assert_eq!(nonzero.get(), count);
            assert_eq!(Budget::from(nonzero), budget);
        }
    }

    #[test]
    fn zero_budget_is_structural_error() {
        assert!(matches!(
            (Budget::try_from(0)).as_ref().unwrap_err().view(),
            crate::SciOxideErrorView::InvalidInput { .. }
        ));
    }

    #[test]
    fn budget_macro_checks_constants() {
        const LIMIT: Budget = crate::budget!(64 * 1024);
        assert_eq!(crate::budget!(1), Budget::ONE);
        assert_eq!(crate::budget!(256).get(), 256);
        assert_eq!(crate::budget!(usize::MAX).get(), usize::MAX);
        assert_eq!(LIMIT.get(), 65_536);
    }

    #[test]
    fn saturating_add_preserves_positive_invariant() {
        assert_eq!(Budget::ONE.saturating_add(99).get(), 100);
        assert_eq!(Budget::ONE.saturating_add(usize::MAX).get(), usize::MAX);
    }
}
