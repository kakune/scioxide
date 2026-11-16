//! Heun's method.

use super::{ButcherTableau, square};
use scioxide_core::RealScalar;

/// Stable real and imaginary axis segments, rounded down.
///
/// R(z) = 1 + z + z²/2 satisfies |R(iy)|² = 1 + y⁴/4 > 1 for every y ≠ 0.
pub(crate) const STABILITY: (f64, f64) = (2.0, 0.0);

explicit_solver!(
/// Integrates with Heun's method, a fixed-step two-stage Runge–Kutta method of
/// order 2 (the explicit trapezoidal rule).
///
/// ```text
/// k₁ = f(tₙ, yₙ),   k₂ = f(tₙ + h, yₙ + h k₁),   yₙ₊₁ = yₙ + (h/2)(k₁ + k₂)
/// ```
///
/// where `h` is the step size. Each step evaluates `f` twice. The method has no
/// error estimate and no dense output, so it supports neither output times nor
/// events. Its stability region contains the real interval `[−2, 0]` and meets the
/// imaginary axis only at zero.
///
/// # References
///
/// - E. Hairer, S. P. Nørsett and G. Wanner, *Solving Ordinary Differential
///   Equations I*, 2nd ed., Springer, 1993, §II.1.
    Heun, 2, 2, false
);

pub(crate) fn tableau<S: RealScalar>() -> ButcherTableau<S, 2> {
    const C: [f64; 2] = [0.0, 1.0];
    const A: [[f64; 2]; 2] = square([&[], &[1.0]]);
    const B: [f64; 2] = [0.5, 0.5];
    ButcherTableau::new(C, A, B)
}

#[cfg(test)]
mod order_tests {
    use super::super::order_conditions::{assert_row_sums, assert_weights};

    #[test]
    fn tableau_has_order_two() {
        let tableau = super::tableau::<f64>();
        assert_row_sums("Heun", &tableau.c, &tableau.a);
        assert_weights("Heun", &tableau.a, 2, &tableau.b);
    }
}
