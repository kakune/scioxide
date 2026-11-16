//! The classical fourth-order Runge–Kutta method.

use super::{ButcherTableau, square};
use scioxide_core::RealScalar;

/// Stable real and imaginary axis segments, rounded down.
///
/// Classical bounds of R(z) = Σ_{k≤4} zᵏ/k!: 2.78529 and 2√2.
pub(crate) const STABILITY: (f64, f64) = (2.7852, 2.8284);

explicit_solver!(
/// Integrates with the classical four-stage Runge–Kutta method of order 4, a
/// fixed-step method.
///
/// ```text
/// k₁ = f(tₙ, yₙ),             k₂ = f(tₙ + h/2, yₙ + (h/2) k₁),
/// k₃ = f(tₙ + h/2, yₙ + (h/2) k₂),   k₄ = f(tₙ + h, yₙ + h k₃),
/// yₙ₊₁ = yₙ + (h/6)(k₁ + 2k₂ + 2k₃ + k₄)
/// ```
///
/// where `h` is the step size. Each step evaluates `f` four times. The method has
/// no error estimate and no dense output, so it supports neither output times nor
/// events. Its stability region contains the real interval `[−2.785, 0]` and the
/// imaginary segment `[−2.828 i, 2.828 i]`.
///
/// # References
///
/// - E. Hairer, S. P. Nørsett and G. Wanner, *Solving Ordinary Differential
///   Equations I*, 2nd ed., Springer, 1993, §II.1.
    ClassicalRk4, 4, 4, false
);

pub(crate) fn tableau<S: RealScalar>() -> ButcherTableau<S, 4> {
    const C: [f64; 4] = [0.0, 0.5, 0.5, 1.0];
    const A: [[f64; 4]; 4] = square([&[], &[0.5], &[0.0, 0.5], &[0.0, 0.0, 1.0]]);
    const B: [f64; 4] = [1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0];
    ButcherTableau::new(C, A, B)
}

#[cfg(test)]
mod order_tests {
    use super::super::order_conditions::{assert_row_sums, assert_weights};

    #[test]
    fn tableau_has_order_four() {
        let tableau = super::tableau::<f64>();
        assert_row_sums("RK4", &tableau.c, &tableau.a);
        assert_weights("RK4", &tableau.a, 4, &tableau.b);
    }
}
