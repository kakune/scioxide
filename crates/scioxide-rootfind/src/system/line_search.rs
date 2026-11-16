//! Backtracking line search with the Armijo sufficient-decrease condition.

use scioxide_core::Scaled;
use scioxide_core::{NumericScalar, RealScalar, ResidualOperator, SciOxideError};

use crate::traits::RootLineSearch;

/// Finds a step length `α` along a descent direction `d` by safeguarded
/// backtracking (Dennis and Schnabel, Algorithm A6.3.1).
///
/// The merit function is `φ(x) = ½ ‖F(x)‖₂²`. Starting from `α = 1`, the search
/// accepts the first `α` that satisfies the Armijo condition
///
/// ```text
/// φ(x + α d) ≤ φ(x) + c α φ′
/// ```
///
/// where `c` is the sufficient-decrease parameter and `φ′ < 0` is the slope of
/// `φ` along `d`. The slope may come from the Jacobian or from a solver's
/// Jacobian model. A rejected `α` is replaced by the minimizer of the quadratic
/// interpolant, and later of the cubic interpolant, of the observed merit
/// values, clamped to `[0.1, 0.5] · α`. A non-finite trial residual, or a
/// `DomainError` with reason `OutsideDomain` from the callback, halves `α` and
/// restarts the interpolation. Other callback errors are returned.
///
/// The full step is always tried. The search gives up and returns `Ok(None)`
/// once the largest relative coordinate move `α · maxᵢ |dᵢ| / max(1, |xᵢ|)`
/// falls below `min_relative_step`. Each trial costs one residual evaluation.
///
/// The defaults are `c = 10⁻⁴` and `min_relative_step = ε^(2/3)`.
///
/// # Examples
///
/// ```
/// use scioxide_core::{SciOxideError, Scaled};
/// use scioxide_rootfind::{BacktrackingLineSearch, ResidualFn, RootLineSearch};
///
/// // F(x) = x at x = 1 with d = −1 and slope φ′ = Fᵀ J d = −1: the full step
/// // reaches the root x = 0.
/// let f = ResidualFn::new(1, |x: &[f64], out: &mut [f64]| {
///     out[0] = x[0];
///     Ok(())
/// });
/// let (mut point, mut residual) = ([0.0], [0.0]);
/// let alpha = BacktrackingLineSearch::default().search_into(
///     &f,
///     &[1.0],
///     &[-1.0],
///     &[1.0],
///     Scaled::from_value(-1.0)?,
///     &mut point,
///     &mut residual,
///     &mut [],
/// )?;
/// assert_eq!(alpha, Some(1.0));
/// assert_eq!((point, residual), ([0.0], [0.0]));
/// # Ok::<(), SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct BacktrackingLineSearch<S: RealScalar> {
    /// Armijo sufficient-decrease parameter (default `10⁻⁴`).
    c: S,
    /// Smallest relative coordinate move before the search gives up (default `ε^(2/3)`).
    min_relative_step: S,
}

impl<S: RealScalar + NumericScalar<Real = S>> Default for BacktrackingLineSearch<S> {
    fn default() -> Self {
        let one: S = <S as scioxide_core::AlgebraicScalar>::ONE;
        let two = one + one;
        let five = two + two + one;
        let ten = five + five;
        Self {
            c: one / (ten * ten * ten * ten),
            min_relative_step: S::CBRT_EPSILON * S::CBRT_EPSILON,
        }
    }
}

impl<S: RealScalar + NumericScalar<Real = S>> RootLineSearch<S> for BacktrackingLineSearch<S> {
    fn search_into(
        &self,
        f: &dyn ResidualOperator<S>,
        x: &[S],
        direction: &[S],
        current_residual: &[S],
        merit_slope: scioxide_core::Scaled<S>,
        trial_point: &mut [S],
        trial_residual: &mut [S],
        scratch: &mut [S],
    ) -> Result<Option<S>, SciOxideError> {
        let dimension = crate::residual::square_dimension(f, scioxide_core::argument!("f"))?;
        for value in [
            x,
            direction,
            current_residual,
            &*trial_point,
            &*trial_residual,
        ] {
            if value.len() != dimension {
                return Err(SciOxideError::dimension_mismatch(
                    scioxide_core::Dimensions::one(dimension),
                    scioxide_core::Dimensions::one(value.len()),
                ));
            }
        }

        let required = f.residual_workspace_len()?;
        let available = scratch.len();
        let scratch = scratch.get_mut(..required).ok_or_else(|| {
            SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::one(required),
                scioxide_core::Dimensions::one(available),
            )
        })?;
        for vector in [x, direction, current_residual] {
            crate::residual::finite(vector, scioxide_core::NonFiniteOrigin::Input)?;
        }
        if merit_slope.cmp_value(Scaled::ZERO) != core::cmp::Ordering::Less {
            return Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::IllConditioned,
                "merit derivative must be strictly negative",
            ));
        }
        let relative = crate::nonlinear_math::relative_length(x, direction);
        let mut search = crate::nonlinear_math::Backtracking::new(merit_slope);
        loop {
            let alpha = search.step();
            crate::nonlinear_math::affine_into(
                trial_point,
                x,
                alpha,
                direction,
                crate::nonlinear_math::StepRounding::Fused,
            )?;
            if trial_point == x {
                return Ok(None);
            }
            let evaluation = f.residual_into(trial_point, trial_residual, scratch);
            match crate::residual::trial_status(evaluation, trial_residual)? {
                crate::residual::TrialStatus::Rejected => search.reject_domain(),
                crate::residual::TrialStatus::Finite => {
                    if search.assess(current_residual, trial_residual, self.c)? {
                        return Ok(Some(alpha));
                    }
                }
            }
            if search.step() * relative < self.min_relative_step {
                return Ok(None);
            }
        }
    }
}

impl<S: RealScalar + NumericScalar<Real = S>> BacktrackingLineSearch<S> {
    /// Creates a line search with sufficient-decrease parameter `c` and step
    /// floor `min_relative_step`.
    ///
    /// The search gives up when the trial move `α · maxᵢ |dᵢ| / max(1, |xᵢ|)`
    /// falls below `min_relative_step`.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `c` or `min_relative_step` is NaN or infinite.
    /// - `InvalidInput` unless `0 < c < 1` and `min_relative_step > 0`.
    pub fn new(c: S, min_relative_step: S) -> Result<Self, SciOxideError> {
        let one: S = <S as scioxide_core::AlgebraicScalar>::ONE;
        let zero: S = <S as scioxide_core::AlgebraicScalar>::ZERO;
        if !c.is_finite() {
            return Err(SciOxideError::non_finite(
                scioxide_core::NonFiniteOrigin::Input,
                "c is non-finite",
            ));
        }
        if c <= zero || c >= one {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("c"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "backtracking line search requires 0 < c < 1",
            ));
        }
        if !min_relative_step.is_finite() {
            return Err(SciOxideError::non_finite(
                scioxide_core::NonFiniteOrigin::Input,
                "min_relative_step is non-finite",
            ));
        }
        if min_relative_step <= zero {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("min_relative_step"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "min_relative_step must be greater than zero",
            ));
        }
        Ok(Self {
            c,
            min_relative_step,
        })
    }

    /// Returns the Armijo sufficient-decrease parameter `c`.
    #[must_use]
    pub const fn c(&self) -> S {
        self.c
    }

    /// Returns the smallest relative coordinate move of a trial step.
    #[must_use]
    pub const fn min_relative_step(&self) -> S {
        self.min_relative_step
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ResidualFn;

    /// `F(x, y) = (x − 1, y − 2)`.
    fn simple_system() -> impl ResidualOperator<f64> {
        ResidualFn::new(2, |x: &[f64], out: &mut [f64]| {
            out[0] = x[0] - 1.0;
            out[1] = x[1] - 2.0;
            Ok(())
        })
    }

    fn search(direction: &[f64]) -> Result<Option<f64>, SciOxideError> {
        BacktrackingLineSearch::<f64>::default().search_into(
            &simple_system(),
            &[0.0, 0.0],
            direction,
            &[-1.0, -2.0],
            Scaled::from_value(-5.0).unwrap(),
            &mut [0.0; 2],
            &mut [0.0; 2],
            &mut [],
        )
    }

    #[test]
    fn test_backtracking_accepts_full_step() {
        let alpha = search(&[1.0, 2.0]).unwrap().unwrap();
        assert!((alpha - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_backtracking_default_params() {
        let ls = BacktrackingLineSearch::<f64>::default();
        assert!(ls.c() > 0.0 && ls.c() < 1.0);
        assert!(ls.min_relative_step() > 0.0);
    }

    #[test]
    fn test_backtracking_rejects_invalid_parameters() {
        assert!(BacktrackingLineSearch::new(1.0, 1e-12).is_err());
        let result = BacktrackingLineSearch::new(1e-4, 0.0);
        assert_eq!(
            result.unwrap_err().kind(),
            scioxide_core::SciOxideErrorKind::InvalidInput
        );
    }

    #[test]
    fn test_backtracking_rejects_invalid_direction_dimension() {
        assert_eq!(
            search(&[1.0]).unwrap_err().kind(),
            scioxide_core::SciOxideErrorKind::DimensionMismatch
        );
    }

    #[test]
    fn non_finite_trial_residuals_exhaust_the_search() {
        let f = ResidualFn::new(1, |_: &[f64], out: &mut [f64]| {
            out[0] = f64::NAN;
            Ok(())
        });
        let step = BacktrackingLineSearch::<f64>::default()
            .search_into(
                &f,
                &[0.0],
                &[1.0],
                &[1.0],
                Scaled::from_value(-1.0).unwrap(),
                &mut [0.0],
                &mut [0.0],
                &mut [],
            )
            .unwrap();
        assert_eq!(step, None);
    }
}
