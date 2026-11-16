//! Verner's 6(5) pair.
//!
//! Dense output is cubic Hermite interpolation, of lower order than the method:
//! interior value error O(h⁴), derivative error O(h³).

use super::{ButcherTableau, square};
use scioxide_core::RealScalar;

/// Stable real and imaginary axis segments, rounded down.
///
/// Computed from the advancing sixth-order solution: 4.06477 and 1.30676.
pub(crate) const STABILITY: (f64, f64) = (4.0647, 1.3067);

explicit_solver!(
/// Integrates with Verner's 6(5) pair, an adaptive eight-stage explicit method.
///
/// The solution advances with the sixth-order formula, and the difference to the
/// embedded fifth-order formula is the local error estimate. Each attempted step
/// evaluates `f` seven times, plus once at each new accepted point; a retried step
/// reuses that evaluation. Dense output is the cubic Hermite interpolant of the
/// step, with interior error `O(h⁴)`. It needs `f(tₙ₊₁, yₙ₊₁)`, which is evaluated
/// only when output times, events or delay history read the step and is then reused
/// by the next step. The stability region contains the real interval `[−4.064, 0]`
/// and the imaginary segment `[−1.306 i, 1.306 i]`.
///
/// # References
///
/// - J. H. Verner, "Explicit Runge–Kutta methods with estimates of the local
///   truncation error", *SIAM J. Numer. Anal.* 15(4), 772–790, 1978.
    Verner65, 8, 6, true
);

pub(crate) fn tableau<S: RealScalar>() -> ButcherTableau<S, 8, 8, true> {
    const C: [f64; 8] = [
        0.0,
        1.0 / 6.0,
        4.0 / 15.0,
        2.0 / 3.0,
        5.0 / 6.0,
        1.0,
        1.0 / 15.0,
        1.0,
    ];
    const A: [[f64; 8]; 8] = square([
        &[],
        &[1.0 / 6.0],
        &[4.0 / 75.0, 16.0 / 75.0],
        &[5.0 / 6.0, -8.0 / 3.0, 5.0 / 2.0],
        &[-165.0 / 64.0, 55.0 / 6.0, -425.0 / 64.0, 85.0 / 96.0],
        &[12.0 / 5.0, -8.0, 4015.0 / 612.0, -11.0 / 36.0, 88.0 / 255.0],
        &[
            -8263.0 / 15000.0,
            124.0 / 75.0,
            -643.0 / 680.0,
            -81.0 / 250.0,
            2484.0 / 10625.0,
            0.0,
        ],
        &[
            3501.0 / 1720.0,
            -300.0 / 43.0,
            297275.0 / 52632.0,
            -319.0 / 2322.0,
            24068.0 / 84065.0,
            0.0,
            3850.0 / 26703.0,
        ],
    ]);
    const B: [f64; 8] = [
        3.0 / 40.0,
        0.0,
        875.0 / 2244.0,
        23.0 / 72.0,
        264.0 / 1955.0,
        0.0,
        125.0 / 11592.0,
        43.0 / 616.0,
    ];
    const LOWER: [f64; 8] = [
        13.0 / 160.0,
        0.0,
        2375.0 / 5984.0,
        5.0 / 16.0,
        12.0 / 85.0,
        3.0 / 44.0,
        0.0,
        0.0,
    ];
    ButcherTableau::new(C, A, B)
        .with_embedded(LOWER, 6)
        .with_dense(
            crate::dense_output::DenseExtension::CubicHermite,
            [[0.0; 8]; 4],
        )
}

#[cfg(test)]
mod order_tests {
    use super::super::order_conditions::{assert_row_sums, assert_weights};

    #[test]
    fn pair_has_orders_six_and_five() {
        let tableau = super::tableau::<f64>();
        assert_row_sums("Verner65", &tableau.c, &tableau.a);
        assert_weights("Verner65", &tableau.a, 6, &tableau.b);
        let embedded = core::array::from_fn(|i| tableau.b[i] - tableau.error[i]);
        assert_weights("Verner65 embedded", &tableau.a, 5, &embedded);
    }
}
