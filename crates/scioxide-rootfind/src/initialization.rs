//! Brackets and starting points for root solvers.

use scioxide_core::{Argument, Interval, RealScalar, SciOxideError};

use crate::traits::finite_output;

/// Holds a finite interval `[a, b]` with the function values `f(a)` and `f(b)`
/// observed at its endpoints.
///
/// Either one endpoint value is zero or the two values have opposite signs, so
/// a continuous `f` has a zero in `[a, b]` by the intermediate value theorem.
/// Only SciOxide creates brackets: [`find_bracket`](crate::find_bracket) returns
/// one, and bracketing solvers report the final bracket in
/// [`RootResult::bracket`](crate::RootResult::bracket).
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct Bracket<S: RealScalar> {
    interval: Interval<S>,
    lower_value: S,
    upper_value: S,
}

/// Outcome of [`Bracket::verify`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BracketVerification {
    /// Both endpoint values equal the stored values.
    Verified,
    /// At least one endpoint value differs from the stored value.
    Rejected,
}

impl<S: RealScalar> Bracket<S> {
    /// Evaluates `function` at both endpoints again and compares the values
    /// with the stored ones.
    ///
    /// This checks the endpoint values, not the continuity of `function` on the
    /// interval. A bracket with `a = b` costs one evaluation; others cost two.
    ///
    /// # Errors
    ///
    /// - `NonFinite` (origin `Callback`) if `function` returns NaN or infinity.
    /// - Errors returned by `function` are returned unchanged.
    pub fn verify<F: crate::ScalarFunction<S> + ?Sized>(
        &self,
        function: &F,
    ) -> Result<BracketVerification, SciOxideError> {
        let lower = finite_output(function.eval(self.lower())?)?;
        let upper = if self.lower() == self.upper() {
            lower
        } else {
            finite_output(function.eval(self.upper())?)?
        };
        Ok(if lower == self.lower_value && upper == self.upper_value {
            BracketVerification::Verified
        } else {
            BracketVerification::Rejected
        })
    }

    /// Returns the lower endpoint `a`.
    #[must_use]
    pub fn lower(&self) -> S {
        self.interval.lower()
    }

    /// Returns the upper endpoint `b`.
    #[must_use]
    pub fn upper(&self) -> S {
        self.interval.upper()
    }

    /// Returns the interval `[a, b]`.
    pub const fn interval(&self) -> Interval<S> {
        self.interval
    }

    /// Returns the observed value `f(a)`.
    #[must_use]
    pub const fn lower_value(&self) -> S {
        self.lower_value
    }

    /// Returns the observed value `f(b)`.
    #[must_use]
    pub const fn upper_value(&self) -> S {
        self.upper_value
    }

    /// Checks finite endpoint values with a sign change or an endpoint zero.
    /// A missing sign change is reported against `argument`.
    pub(crate) fn from_evaluated(
        interval: Interval<S>,
        lower_value: S,
        upper_value: S,
        argument: Argument,
    ) -> Result<Self, SciOxideError> {
        finite_output(lower_value)?;
        finite_output(upper_value)?;
        validate_sign_change(lower_value, upper_value, argument)?;
        Ok(Self {
            interval,
            lower_value,
            upper_value,
        })
    }
}

impl<S: RealScalar> From<Bracket<S>> for Interval<S> {
    fn from(bracket: Bracket<S>) -> Self {
        bracket.interval
    }
}

/// Requires `value` to hold `dimension` finite components: a
/// `DimensionMismatch` for another length and a `NonFinite` error (origin
/// `Input`) for a NaN or infinite component.
pub(crate) fn validate_initial<S: RealScalar>(
    dimension: usize,
    value: &[S],
) -> Result<(), SciOxideError> {
    if value.len() != dimension {
        return Err(SciOxideError::dimension_mismatch(
            [dimension],
            [value.len()],
        ));
    }
    if let Some(index) = value.iter().position(|component| !component.is_finite()) {
        return Err(SciOxideError::non_finite(
            scioxide_core::NonFiniteOrigin::Input,
            alloc::format!("initial component {index} is non-finite"),
        ));
    }
    Ok(())
}

/// Whether `left` or `right` is zero or they have opposite signs. Brackets,
/// bracket search and bracketing solvers all use this test.
///
/// Values are finite, so this is `left·right <= 0` without the product's
/// underflow. The comparisons combine without short-circuiting: a bisection's
/// side choice is unpredictable, and one select avoids a chain of branches.
#[inline(always)]
pub(crate) fn brackets_root<S: RealScalar>(left: S, right: S) -> bool {
    let zero = S::ZERO;
    ((left <= zero) & (right >= zero)) | ((left >= zero) & (right <= zero))
}

/// Requires a sign change or an endpoint zero, reporting `argument` otherwise.
pub(crate) fn validate_sign_change<S: RealScalar>(
    left: S,
    right: S,
    argument: Argument,
) -> Result<(), SciOxideError> {
    if brackets_root(left, right) {
        Ok(())
    } else {
        Err(SciOxideError::invalid_input(
            argument,
            scioxide_core::InvalidInputReason::Inconsistent,
            "bracket endpoint values must have opposite signs or contain an endpoint root",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{NewtonSystem, NewtonSystemOptions, SystemRootResult, SystemRootSolver};
    use scioxide_core::{ResidualOperator, SciOxideErrorKind, SciOxideErrorView};

    const BRACKET: Argument = scioxide_core::argument!("bracket");

    /// A residual with `inputs` unknowns and `outputs` equations.
    struct Shape {
        inputs: usize,
        outputs: usize,
    }

    impl ResidualOperator<f64> for Shape {
        fn input_dimension(&self) -> usize {
            self.inputs
        }
        fn output_dimension(&self) -> usize {
            self.outputs
        }
        fn residual_workspace_len(&self) -> Result<usize, SciOxideError> {
            Ok(0)
        }
        fn residual_into(
            &self,
            x: &[f64],
            out: &mut [f64],
            _: &mut [f64],
        ) -> Result<(), SciOxideError> {
            out.copy_from_slice(x);
            Ok(())
        }
    }
    impl crate::DenseJacobian<f64> for Shape {}

    const TWO_DIMENSIONAL: Shape = Shape {
        inputs: 2,
        outputs: 2,
    };

    #[test]
    fn solver_built_bracket_retains_values_and_accepts_endpoint_root() {
        let interval = Interval::new(1.0, 2.0).unwrap();
        let bracket = Bracket::from_evaluated(interval, 0.0, 1.0, BRACKET).unwrap();
        assert_eq!((bracket.lower_value(), bracket.upper_value()), (0.0, 1.0));
        let point_root =
            Bracket::from_evaluated(Interval::new(1.0, 1.0).unwrap(), 0.0, 0.0, BRACKET).unwrap();
        assert_eq!((point_root.lower(), point_root.upper()), (1.0, 1.0));
    }

    #[test]
    fn bracket_rejects_same_sign_values() {
        let result = Bracket::from_evaluated(Interval::new(2.0, 3.0).unwrap(), 1.0, 2.0, BRACKET);
        assert_eq!(
            result.unwrap_err().kind(),
            scioxide_core::SciOxideErrorKind::InvalidInput
        );
    }

    #[test]
    fn bracket_rejects_non_finite_callback_value() {
        let result =
            Bracket::from_evaluated(Interval::new(1.0, 2.0).unwrap(), f64::NAN, 1.0, BRACKET);
        assert_eq!(
            result.unwrap_err().kind(),
            scioxide_core::SciOxideErrorKind::NonFinite
        );
    }

    /// Solves `function` from `initial` with Newton's method and default options.
    fn solve(function: &Shape, initial: &[f64]) -> Result<SystemRootResult<f64>, SciOxideError> {
        NewtonSystem::new(NewtonSystemOptions::default())?.solve(function, initial)
    }

    #[test]
    fn system_solve_validates_dimension_and_finiteness() {
        let valid = solve(&TWO_DIMENSIONAL, &[1.0, 2.0]).unwrap();
        assert!(valid.is_success());
        assert!(valid.point().iter().all(|x| x.abs() < 1e-7));

        let mismatch = solve(&TWO_DIMENSIONAL, &[1.0]);
        assert_eq!(
            mismatch.unwrap_err().kind(),
            SciOxideErrorKind::DimensionMismatch
        );

        let non_finite = solve(&TWO_DIMENSIONAL, &[1.0, f64::INFINITY]);
        assert!(matches!(
            non_finite.unwrap_err().view(),
            SciOxideErrorView::NonFinite {
                origin: scioxide_core::NonFiniteOrigin::Input,
                ..
            }
        ));

        let rectangular = Shape {
            inputs: 2,
            outputs: 3,
        };
        assert_eq!(
            solve(&rectangular, &[1.0, 2.0]).unwrap_err().kind(),
            SciOxideErrorKind::DimensionMismatch
        );
        let empty = Shape {
            inputs: 0,
            outputs: 0,
        };
        assert!(matches!(
            solve(&empty, &[]).unwrap_err().view(),
            SciOxideErrorView::InvalidInput { argument, .. }
                if argument == scioxide_core::argument!("function")
        ));
    }
}
