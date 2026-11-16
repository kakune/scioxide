//! Fixed-point iteration and residual mixing with bounded history.
use crate::RootTermination;
use crate::invalid;
use alloc::vec::Vec;
use core::ops::Neg;
use scioxide_core::AlgebraicScalar;
use scioxide_core::{
    Argument, Budget, EvaluationCounts, EvaluationRole, ExecutionPolicy, InvalidInputReason,
    NonFiniteOrigin, NumericScalar, RealScalar, ResidualOperator, ResidualReport,
    ResidualTolerance, SciOxideError, SolveContext, StepTolerance,
};
use scioxide_core::{Scaled, ScaledValue};
use scioxide_linalg::{
    HouseholderDecomposition, HouseholderOrientation, LinalgWorkspace, Matrix, RankPolicy, Vector,
};

mod kernels;
mod options;
mod plan;
mod result;
mod workspace;
#[cfg(test)]
use kernels::*;
pub use options::{FixedPointInput, FixedPointMethod, FixedPointOptions, HistoryRestart};
pub use plan::{FixedPointSolver, fixed_point};
pub use result::{FixedPointDiagnostics, FixedPointResult};
pub use workspace::{FixedPointWorkspace, FixedPointWorkspaceRequirements};

/// Require a positive finite `argument`.
fn positive<R: RealScalar>(value: R, argument: Argument) -> Result<(), SciOxideError> {
    if value.is_finite() && value > R::ZERO {
        Ok(())
    } else {
        Err(invalid(
            argument,
            InvalidInputReason::Inconsistent,
            "expected a positive finite value",
        ))
    }
}
use crate::nonlinear_math::{affine, affine_scaled, difference, dot, norm, rounded, scaled};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_history_reduction_retains_small_cancellation_term() {
        let value = dot(&[1e300_f64, 1.0, -1e300], &[1.0; 3])
            .unwrap()
            .round()
            .unwrap();
        assert_eq!(value, 1.0);
    }
    #[test]
    fn balanced_secant_factors_preserve_extreme_finite_update() {
        let mut row = [1e-300_f64];
        let mut correction = [0.0];
        assert!(broyden_correction(&[1e300], &[0.0], &mut row, &[1.0], &mut correction).unwrap());
        let entry = affine(0.0, correction[0], row[0]).unwrap();
        assert!((entry / 1e300 - 1.0).abs() < 1e-14);
    }

    #[test]
    fn balanced_factors_replay_tiny_components_before_amplification() {
        let mut row = [1e300_f64, 0.0];
        let mut correction = [0.0; 2];
        assert!(
            broyden_correction(
                &[1e300, 1e-300],
                &[0.0; 2],
                &mut row,
                &[1e-300, 0.0],
                &mut correction
            )
            .unwrap()
        );
        assert!((correction[0] / 1e300 - 1.0).abs() < 1e-14);
        assert!((correction[1] / 1e-300 - 1.0).abs() < 1e-14);
    }

    #[test]
    fn component_division_rounds_after_the_norm() {
        let tiny = f64::from_bits(1);
        let options = FixedPointOptions::default().with_component_scales(alloc::vec![2.0; 2]);
        let value = component_norm(&[3.0 * tiny; 2], &options).unwrap();
        assert_eq!(value, 2.0 * tiny);
    }

    #[test]
    fn history_contractions_replay_before_final_projection() {
        fn check<T: NumericScalar>() {
            let max = T::from_real(T::Real::MAX);
            let half = T::from_real(T::Real::HALF);
            let quarter = half * half;
            let value =
                subtract_history(T::ZERO, T::ONE, [(max, max, quarter)].into_iter()).unwrap();
            assert!(
                (value.real_component() / T::Real::MAX + T::Real::HALF).abs() <= T::Real::EPSILON
            );
            let two = T::ONE + T::ONE;
            let value = subtract_history(
                max,
                T::ONE,
                [(max, T::ZERO, -two), (max, T::ZERO, two)].into_iter(),
            )
            .unwrap();
            assert!(
                (value.real_component() / T::Real::MAX - T::Real::ONE).abs() <= T::Real::EPSILON
            );
        }
        check::<f32>();
        check::<f64>();
        check::<scioxide_core::Complex<f32>>();
        check::<scioxide_core::Complex<f64>>();
    }
}
