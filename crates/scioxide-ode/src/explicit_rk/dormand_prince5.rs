//! The Dormand–Prince 5(4) pair.
//!
//! The last stage of a step evaluates `f` at the new solution and is the first stage
//! of the next step (first same as last, FSAL).

use super::{ButcherTableau, square};
use scioxide_core::RealScalar;

/// Stable real and imaginary axis segments, rounded down.
///
/// R(z) = Σ_{k≤5} zᵏ/k! + z⁶/600: real bound 3.30656, imaginary bound 0.99718.
pub(crate) const STABILITY: (f64, f64) = (3.3065, 0.9971);

explicit_solver!(
/// Integrates with the Dormand–Prince 5(4) pair, an adaptive seven-stage explicit
/// method and the usual first choice for nonstiff problems.
///
/// The solution advances with the fifth-order formula, and the difference to the
/// embedded fourth-order formula is the local error estimate. The last stage
/// evaluates `f` at the new solution and is reused as the first stage of the next
/// step, so each step after the first evaluates `f` six times. Dense output is
/// Shampine's fourth-order continuous extension, which needs no extra evaluations.
/// The stability region contains the real interval `[−3.306, 0]` and the imaginary
/// segment `[−0.997 i, 0.997 i]`.
///
/// # References
///
/// - J. R. Dormand and P. J. Prince, "A family of embedded Runge–Kutta formulae",
///   *J. Comput. Appl. Math.* 6(1), 19–26, 1980.
/// - L. F. Shampine, "Some practical Runge–Kutta formulas", *Math. Comp.* 46(173),
///   135–150, 1986.
    DormandPrince5, 7, 5, true
);

pub(crate) fn tableau<S: RealScalar>() -> ButcherTableau<S, 7, 7, true> {
    const C: [f64; 7] = [0.0, 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0];
    const A: [[f64; 7]; 7] = square([
        &[],
        &[1.0 / 5.0],
        &[3.0 / 40.0, 9.0 / 40.0],
        &[44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0],
        &[
            19372.0 / 6561.0,
            -25360.0 / 2187.0,
            64448.0 / 6561.0,
            -212.0 / 729.0,
        ],
        &[
            9017.0 / 3168.0,
            -355.0 / 33.0,
            46732.0 / 5247.0,
            49.0 / 176.0,
            -5103.0 / 18656.0,
        ],
        &[
            35.0 / 384.0,
            0.0,
            500.0 / 1113.0,
            125.0 / 192.0,
            -2187.0 / 6784.0,
            11.0 / 84.0,
        ],
    ]);
    const B: [f64; 7] = [
        35.0 / 384.0,
        0.0,
        500.0 / 1113.0,
        125.0 / 192.0,
        -2187.0 / 6784.0,
        11.0 / 84.0,
        0.0,
    ];
    const LOWER: [f64; 7] = [
        5179.0 / 57600.0,
        0.0,
        7571.0 / 16695.0,
        393.0 / 640.0,
        -92097.0 / 339200.0,
        187.0 / 2100.0,
        1.0 / 40.0,
    ];
    ButcherTableau::new(C, A, B)
        .with_embedded(LOWER, 5)
        .with_dense(
            crate::dense_output::DenseExtension::DormandPrince5,
            [DENSE_WEIGHTS, [0.0; 7], [0.0; 7], [0.0; 7]],
        )
        .with_endpoint(super::Endpoint::Stage(6))
}

/// The coefficients of Shampine's fourth-order continuous extension, in the
/// alternating basis.
pub(crate) const DENSE_WEIGHTS: [f64; 7] = [
    -12715105075.0 / 11282082432.0,
    0.0,
    87487479700.0 / 32700410799.0,
    -10690763975.0 / 1880347072.0,
    701980252875.0 / 199316789632.0,
    -1453857185.0 / 822651844.0,
    69997945.0 / 29380423.0,
];

#[cfg(test)]
mod order_tests {
    use super::super::order_conditions::{assert_dense, assert_row_sums, assert_weights};

    #[test]
    fn pair_and_continuous_extension_have_their_orders() {
        let tableau = super::tableau::<f64>();
        assert_row_sums("DP5", &tableau.c, &tableau.a);
        assert_weights("DP5", &tableau.a, 5, &tableau.b);
        let embedded = core::array::from_fn(|i| tableau.b[i] - tableau.error[i]);
        assert_weights("DP5 embedded", &tableau.a, 4, &embedded);
        // First same as last: the seventh stage evaluates f at the advanced solution.
        assert_eq!(tableau.c[6], 1.0);
        assert_eq!(tableau.a[6], tableau.b);
        assert_dense("DP5 dense", &tableau.a, 4, |theta| {
            super::super::order_conditions::alternating(&tableau.b, 6, &tableau.dense[..1], theta)
        });
    }
}
