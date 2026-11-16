//! The Runge–Kutta–Fehlberg 5(4) pair.
//!
//! Dense output is cubic Hermite interpolation, of lower order than the method:
//! interior value error O(h⁴), derivative error O(h³).

use super::{ButcherTableau, square};
use scioxide_core::RealScalar;

/// Stable real and imaginary axis segments, rounded down.
///
/// The advancing fifth-order solution is unstable on the imaginary axis near zero.
pub(crate) const STABILITY: (f64, f64) = (3.6777, 0.0);

explicit_solver!(
/// Integrates with the Runge–Kutta–Fehlberg 5(4) pair, an adaptive six-stage
/// explicit method.
///
/// The solution advances with the fifth-order formula (local extrapolation), and
/// the difference to the embedded fourth-order formula is the local error
/// estimate. Each attempted step evaluates `f` five times, plus once at each new
/// accepted point; a retried step reuses that evaluation. Dense output is the cubic
/// Hermite interpolant of the step, with interior error `O(h⁴)`. It needs
/// `f(tₙ₊₁, yₙ₊₁)`, which is evaluated only when output times, events or delay
/// history read the step and is then reused by the next step. The stability region
/// contains the real interval `[−3.677, 0]` and meets the imaginary axis only at
/// zero.
///
/// # References
///
/// - E. Fehlberg, "Low-order classical Runge–Kutta formulas with stepsize control
///   and their application to some heat transfer problems", NASA Technical
///   Report R-315, 1969.
    Rkf45, 6, 5, true
);

pub(crate) fn tableau<S: RealScalar>() -> ButcherTableau<S, 6, 6, true> {
    const C: [f64; 6] = [0.0, 1.0 / 4.0, 3.0 / 8.0, 12.0 / 13.0, 1.0, 1.0 / 2.0];
    const A: [[f64; 6]; 6] = square([
        &[],
        &[1.0 / 4.0],
        &[3.0 / 32.0, 9.0 / 32.0],
        &[1932.0 / 2197.0, -7200.0 / 2197.0, 7296.0 / 2197.0],
        &[439.0 / 216.0, -8.0, 3680.0 / 513.0, -845.0 / 4104.0],
        &[
            -8.0 / 27.0,
            2.0,
            -3544.0 / 2565.0,
            1859.0 / 4104.0,
            -11.0 / 40.0,
        ],
    ]);
    const B: [f64; 6] = [
        16.0 / 135.0,
        0.0,
        6656.0 / 12825.0,
        28561.0 / 56430.0,
        -9.0 / 50.0,
        2.0 / 55.0,
    ];
    const LOWER: [f64; 6] = [
        25.0 / 216.0,
        0.0,
        1408.0 / 2565.0,
        2197.0 / 4104.0,
        -1.0 / 5.0,
        0.0,
    ];
    ButcherTableau::new(C, A, B)
        .with_embedded(LOWER, 5)
        .with_dense(
            crate::dense_output::DenseExtension::CubicHermite,
            [[0.0; 6]; 4],
        )
}

#[cfg(test)]
mod order_tests {
    use super::super::order_conditions::{assert_row_sums, assert_weights};

    #[test]
    fn pair_has_orders_five_and_four() {
        let tableau = super::tableau::<f64>();
        assert_row_sums("RKF45", &tableau.c, &tableau.a);
        assert_weights("RKF45", &tableau.a, 5, &tableau.b);
        let embedded = core::array::from_fn(|i| tableau.b[i] - tableau.error[i]);
        assert_weights("RKF45 embedded", &tableau.a, 4, &embedded);
    }
}
