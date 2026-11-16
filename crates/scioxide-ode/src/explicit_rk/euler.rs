//! The forward Euler method.

use super::{ButcherTableau, square};
use scioxide_core::RealScalar;

/// Stable real and imaginary axis segments, rounded down.
///
/// Forward Euler is stable on the disk |1 + z| ≤ 1, which touches the imaginary axis only at zero.
pub(crate) const STABILITY: (f64, f64) = (2.0, 0.0);

explicit_solver!(
/// Integrates with the forward Euler method, a fixed-step explicit method of order 1.
///
/// ```text
/// yₙ₊₁ = yₙ + h f(tₙ, yₙ)
/// ```
///
/// where `h` is the step size. Each step evaluates `f` once. The method has no
/// error estimate and no dense output, so it supports neither output times nor
/// events. It is stable for `h λ` in the disk `|1 + hλ| ≤ 1`, which meets the real
/// axis in `[−2, 0]` and the imaginary axis only at zero.
///
/// # References
///
/// - E. Hairer, S. P. Nørsett and G. Wanner, *Solving Ordinary Differential
///   Equations I*, 2nd ed., Springer, 1993, §II.1.
    ForwardEuler, 1, 1, false
);

pub(crate) fn tableau<S: RealScalar>() -> ButcherTableau<S, 1> {
    const C: [f64; 1] = [0.0];
    const A: [[f64; 1]; 1] = square([&[]]);
    const B: [f64; 1] = [1.0];
    ButcherTableau::new(C, A, B)
}

#[cfg(test)]
mod order_tests {
    use super::super::order_conditions::{assert_row_sums, assert_weights};

    #[test]
    fn tableau_has_order_one() {
        let tableau = super::tableau::<f64>();
        assert_row_sums("Euler", &tableau.c, &tableau.a);
        assert_weights("Euler", &tableau.a, 1, &tableau.b);
    }
}
