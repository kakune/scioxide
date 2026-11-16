//! The Dormand–Prince 8(5,3) method with its seventh-order continuous extension.
// Coefficient tables: Copyright (c) 2004, Ernst Hairer; BSD-2-Clause.
// See THIRD_PARTY_NOTICES.md for the source and complete redistribution terms.

#![allow(clippy::excessive_precision, clippy::unreadable_literal)]

use super::{ButcherTableau, Endpoint, square};
use scioxide_core::{NumericScalar, RealScalar, SciOxideError};

const DOP853_STAGES: usize = 12;

const DOP853_NODES: [f64; DOP853_STAGES] = [
    0.0,
    0.526001519587677318785587544488e-1,
    0.789002279381515978178381316732e-1,
    0.118350341907227396726757197510,
    0.281649658092772603273242802490,
    0.333333333333333333333333333333,
    0.25,
    0.307692307692307692307692307692,
    0.651282051282051282051282051282,
    0.6,
    0.857142857142857142857142857142,
    1.0,
];

const DOP853_WEIGHTS: [f64; DOP853_STAGES] = [
    5.42937341165687622380535766363e-2,
    0.0,
    0.0,
    0.0,
    0.0,
    4.45031289275240888144113950566,
    1.89151789931450038304281599044,
    -5.8012039600105847814672114227,
    3.1116436695781989440891606237e-1,
    -1.52160949662516078556178806805e-1,
    2.01365400804030348374776537501e-1,
    4.47106157277725905176885569043e-2,
];

const DOP853_ERROR_WEIGHTS: [f64; DOP853_STAGES] = [
    0.1312004499419488073250102996e-1,
    0.0,
    0.0,
    0.0,
    0.0,
    -0.1225156446376204440720569753e1,
    -0.4957589496572501915214079952,
    0.1664377182454986536961530415e1,
    -0.3503288487499736816886487290,
    0.3341791187130174790297318841,
    0.8192320648511571246570742613e-1,
    -0.2235530786388629525884427845e-1,
];

const DOP853_STAGE_WEIGHTS: [&[f64]; DOP853_STAGES] = [
    &[],
    &[5.26001519587677318785587544488e-2],
    &[
        1.97250569845378994544595329183e-2,
        5.91751709536136983633785987549e-2,
    ],
    &[
        2.95875854768068491816892993775e-2,
        0.0,
        8.87627564304205475450678981324e-2,
    ],
    &[
        2.41365134159266685502369798665e-1,
        0.0,
        -8.84549479328286085344864962717e-1,
        9.24834003261792003115737966543e-1,
    ],
    &[
        3.7037037037037037037037037037e-2,
        0.0,
        0.0,
        1.70828608729473871279604482173e-1,
        1.25467687566822425016691814123e-1,
    ],
    &[
        3.7109375e-2,
        0.0,
        0.0,
        1.70252211019544039314978060272e-1,
        6.02165389804559606850219397283e-2,
        -1.7578125e-2,
    ],
    &[
        3.70920001185047927108779319836e-2,
        0.0,
        0.0,
        1.70383925712239993810214054705e-1,
        1.07262030446373284651809199168e-1,
        -1.53194377486244017527936158236e-2,
        8.27378916381402288758473766002e-3,
    ],
    &[
        6.24110958716075717114429577812e-1,
        0.0,
        0.0,
        -3.36089262944694129406857109825,
        -8.68219346841726006818189891453e-1,
        2.75920996994467083049415600797e1,
        2.01540675504778934086186788979e1,
        -4.34898841810699588477366255144e1,
    ],
    &[
        4.77662536438264365890433908527e-1,
        0.0,
        0.0,
        -2.48811461997166764192642586468,
        -5.90290826836842996371446475743e-1,
        2.12300514481811942347288949897e1,
        1.52792336328824235832596922938e1,
        -3.32882109689848629194453265587e1,
        -2.03312017085086261358222928593e-2,
    ],
    &[
        -9.3714243008598732571704021658e-1,
        0.0,
        0.0,
        5.18637242884406370830023853209,
        1.09143734899672957818500254654,
        -8.14978701074692612513997267357,
        -1.85200656599969598641566180701e1,
        2.27394870993505042818970056734e1,
        2.49360555267965238987089396762,
        -3.0467644718982195003823669022,
    ],
    &[
        2.27331014751653820792359768449,
        0.0,
        0.0,
        -1.05344954667372501984066689879e1,
        -2.00087205822486249909675718444,
        -1.79589318631187989172765950534e1,
        2.79488845294199600508499808837e1,
        -2.85899827713502369474065508674,
        -8.87285693353062954433549289258,
        1.23605671757943030647266201528e1,
        6.43392746015763530355970484046e-1,
    ],
];

/// Stable real and imaginary axis segments, rounded down.
///
/// Computed from the twelve advancing stages: 6.39365 and 5.96035.
pub(crate) const STABILITY: (f64, f64) = (6.3936, 5.9603);

explicit_solver!(
/// Integrates with the Dormand–Prince 8(5,3) method, an adaptive twelve-stage
/// explicit method of order 8 for smooth problems at tight tolerances.
///
/// The local error estimate combines a fifth-order and a third-order embedded
/// estimate. The first step evaluates `f` thirteen times and each later step
/// twelve, because `f(tₙ₊₁, yₙ₊₁)` is reused as the first stage of the next step;
/// a rejected step keeps its first stage. Dense output is a seventh-order
/// continuous extension with three more stages, evaluated only when output times,
/// events or delay history read the step. The stability region contains the real
/// interval `[−6.393, 0]` and the imaginary segment `[−5.960 i, 5.960 i]`.
///
/// # References
///
/// - E. Hairer, S. P. Nørsett and G. Wanner, *Solving Ordinary Differential
///   Equations I*, 2nd ed., Springer, 1993.
/// - P. J. Prince and J. R. Dormand, "High order embedded Runge–Kutta formulae",
///   *J. Comput. Appl. Math.* 7(1), 67–75, 1981.
    DormandPrince853, 16, 8, true, 12, true
);

pub(crate) fn tableau<S: RealScalar>() -> ButcherTableau<S, 16, 12, true, true> {
    const NODES: [f64; 16] = {
        let mut nodes = [1.0; 16];
        let mut i = 0;
        while i < 12 {
            nodes[i] = DOP853_NODES[i];
            i += 1;
        }
        i = 0;
        while i < 3 {
            nodes[13 + i] = DENSE_NODES[i];
            i += 1;
        }
        nodes
    };
    const A: [[f64; 16]; 16] = square([
        DOP853_STAGE_WEIGHTS[0],
        DOP853_STAGE_WEIGHTS[1],
        DOP853_STAGE_WEIGHTS[2],
        DOP853_STAGE_WEIGHTS[3],
        DOP853_STAGE_WEIGHTS[4],
        DOP853_STAGE_WEIGHTS[5],
        DOP853_STAGE_WEIGHTS[6],
        DOP853_STAGE_WEIGHTS[7],
        DOP853_STAGE_WEIGHTS[8],
        DOP853_STAGE_WEIGHTS[9],
        DOP853_STAGE_WEIGHTS[10],
        DOP853_STAGE_WEIGHTS[11],
        &DOP853_WEIGHTS,
        DENSE_STAGES[0],
        DENSE_STAGES[1],
        DENSE_STAGES[2],
    ]);
    const fn padded(weights: [f64; 12]) -> [f64; 16] {
        let mut row = [0.0; 16];
        let mut i = 0;
        while i < 12 {
            row[i] = weights[i];
            i += 1;
        }
        row
    }
    ButcherTableau::new(NODES, A, padded(DOP853_WEIGHTS))
        .with_error(padded(DOP853_ERROR_WEIGHTS).map(S::from_f64_ieee), 8)
        .with_dense(
            crate::dense_output::DenseExtension::DormandPrince853,
            DENSE_WEIGHTS,
        )
        .with_endpoint(Endpoint::Evaluate)
        .with_correction(padded(DOP853_ERROR3_WEIGHTS).map(S::from_f64_ieee))
}

// Continuous-extension coefficients from Hairer/Wanner DOP853 (CONTD8).
// https://www.unige.ch/~hairer/prog/nonstiff/dop853.f
const DENSE_NODES: [f64; 3] = [0.1, 0.2, 0.777777777777777777777777777778];
const DENSE_STAGES: [&[f64]; 3] = [
    &[
        5.61675022830479523392909219681e-2,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        2.53500210216624811088794765333e-1,
        -2.46239037470802489917441475441e-1,
        -1.24191423263816360469010140626e-1,
        1.5329179827876569731206322685e-1,
        8.20105229563468988491666602057e-3,
        7.56789766054569976138603589584e-3,
        -8.298e-3,
    ],
    &[
        3.18346481635021405060768473261e-2,
        0.0,
        0.0,
        0.0,
        0.0,
        2.83009096723667755288322961402e-2,
        5.35419883074385676223797384372e-2,
        -5.49237485713909884646569340306e-2,
        0.0,
        0.0,
        -1.08347328697249322858509316994e-4,
        3.82571090835658412954920192323e-4,
        -3.40465008687404560802977114492e-4,
        1.41312443674632500278074618366e-1,
    ],
    &[
        -4.28896301583791923408573538692e-1,
        0.0,
        0.0,
        0.0,
        0.0,
        -4.69762141536116384314449447206,
        7.68342119606259904184240953878,
        4.06898981839711007970213554331,
        3.56727187455281109270669543021e-1,
        0.0,
        0.0,
        0.0,
        -1.39902416515901462129418009734e-3,
        2.9475147891527723389556272149,
        -9.15095847217987001081870187138,
    ],
];
const DENSE_WEIGHTS: [[f64; 16]; 4] = [
    [
        -0.84289382761090128651353491142e+1,
        0.0,
        0.0,
        0.0,
        0.0,
        0.56671495351937776962531783590,
        -0.30689499459498916912797304727e+1,
        0.23846676565120698287728149680e+1,
        0.21170345824450282767155149946e+1,
        -0.87139158377797299206789907490,
        0.22404374302607882758541771650e+1,
        0.63157877876946881815570249290,
        -0.88990336451333310820698117400e-1,
        0.18148505520854727256656404962e+2,
        -0.91946323924783554000451984436e+1,
        -0.44360363875948939664310572000e+1,
    ],
    [
        0.10427508642579134603413151009e+2,
        0.0,
        0.0,
        0.0,
        0.0,
        0.24228349177525818288430175319e+3,
        0.16520045171727028198505394887e+3,
        -0.37454675472269020279518312152e+3,
        -0.22113666853125306036270938578e+2,
        0.77334326684722638389603898808e+1,
        -0.30674084731089398182061213626e+2,
        -0.93321305264302278729567221706e+1,
        0.15697238121770843886131091075e+2,
        -0.31139403219565177677282850411e+2,
        -0.93529243588444783865713862664e+1,
        0.35816841486394083752465898540e+2,
    ],
    [
        0.19985053242002433820987653617e+2,
        0.0,
        0.0,
        0.0,
        0.0,
        -0.38703730874935176555105901742e+3,
        -0.18917813819516756882830838328e+3,
        0.52780815920542364900561016686e+3,
        -0.11573902539959630126141871134e+2,
        0.68812326946963000169666922661e+1,
        -0.10006050966910838403183860980e+1,
        0.77771377980534432092869265740,
        -0.27782057523535084065932004339e+1,
        -0.60196695231264120758267380846e+2,
        0.84320405506677161018159903784e+2,
        0.11992291136182789328035130030e+2,
    ],
    [
        -0.25693933462703749003312586129e+2,
        0.0,
        0.0,
        0.0,
        0.0,
        -0.15418974869023643374053993627e+3,
        -0.23152937917604549567536039109e+3,
        0.35763911791061412378285349910e+3,
        0.93405324183624310003907691704e+2,
        -0.37458323136451633156875139351e+2,
        0.10409964950896230045147246184e+3,
        0.29840293426660503123344363579e+2,
        -0.43533456590011143754432175058e+2,
        0.96324553959188282948394950600e+2,
        -0.39177261675615439165231486172e+2,
        -0.14972683625798562581422125276e+3,
    ],
];

// E3 = B - (BHH1 at stage 1, BHH2 at stage 9, BHH3 at stage 12).
const DOP853_ERROR3_WEIGHTS: [f64; DOP853_STAGES] = {
    let mut weights = DOP853_WEIGHTS;
    weights[0] -= 0.244094488188976377952755905512;
    weights[8] -= 0.733846688281611857341361741547;
    weights[11] -= 0.220588235294117647058823529412e-1;
    weights
};

#[inline]
fn scale_error<S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S>>(
    step: S,
    correction: S,
    estimate: T,
) -> Result<T, SciOxideError> {
    let factor = step * correction;
    if factor.is_normal() {
        let value = T::from_components(
            factor * estimate.real_component(),
            if T::IS_REAL {
                S::ZERO
            } else {
                factor * estimate.imaginary_component()
            },
        );
        if value.has_finite_components() {
            return Ok(value);
        }
    }
    // A tiny/overflowing factor may regain range when multiplied by estimate.
    crate::safety::stage_affine(T::ZERO, step, [(correction, estimate)])
}

pub(super) fn combined_error<
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S>,
>(
    dt: S,
    e5: T,
    e3: T,
) -> Result<T, SciOxideError> {
    if e5 == T::ZERO {
        return Ok(T::ZERO);
    }
    let magnitude = e5.abs();
    let third = e3.abs();
    let tenth = S::try_from_f64_lossy(0.1)?;
    // Ordinary estimates need one ratio. Squared-range guards retain the
    // normalized replay for extreme magnitudes and premature underflow.
    let fifth_squared = magnitude * magnitude;
    let scaled_third = tenth * third;
    let norm_squared = fifth_squared + scaled_third * scaled_third;
    if fifth_squared.is_normal() && norm_squared.is_normal() {
        let correction = magnitude / norm_squared.sqrt();
        if correction.is_normal() {
            return scale_error(dt, correction, e5);
        }
    }
    let scale = magnitude.max(third);
    if scale == S::ZERO {
        return Ok(T::ZERO);
    }
    let a = magnitude / scale;
    let b = tenth * (third / scale);
    // Both normalized coordinates are at most one, and one is at least 0.1.
    // Squaring therefore cannot lose the norm's range or significant bits.
    let denominator = (a * a + b * b).sqrt();
    let correction = a / denominator;
    if correction.is_normal() {
        return scale_error(dt, correction, e5);
    }
    crate::safety::StageValue::from_value(dt)?
        .mul(crate::safety::StageValue::from_value(e5)?)?
        .mul(crate::safety::StageValue::from_value(magnitude)?)?
        .div(crate::safety::StageValue::from_value(scale)?)?
        .div(crate::safety::StageValue::from_value(a.hypot(b))?)?
        .round()
}

#[cfg(test)]
mod tests {
    use super::*;
    use scioxide_linalg::Vector;

    fn assert_consistent_sum(values: &[f64], expected: f64) {
        let actual = values.iter().sum::<f64>();
        let scale = values
            .iter()
            .map(|value| value.abs())
            .sum::<f64>()
            .max(expected.abs())
            .max(1.0);
        let bound = 8.0 * f64::EPSILON * scale;
        assert!(
            (actual - expected).abs() <= bound,
            "inconsistent coefficient sum: actual={actual:e}, expected={expected:e}, bound={bound:e}"
        );
    }

    #[test]
    fn error_scaling_retains_range_before_its_final_product() {
        let restored = scale_error(1e-308_f64, 1e-100, 1e308).unwrap();
        assert!((restored / 1e-100 - 1.0).abs() <= 8.0 * f64::EPSILON);
        let tiny = f64::from_bits(1);
        let expected = (tiny * f64::MAX) * 0.5;
        let restored = scale_error(f64::MAX, 0.5, tiny).unwrap();
        assert!((restored - expected).abs() <= 8.0 * f64::EPSILON * expected);
    }

    #[test]
    fn ordinary_error_estimates_match_the_embedded_pair_formula() {
        for sign in [-1.0_f64, 1.0] {
            let actual = combined_error(0.25, sign * 3.0, 4.0).unwrap();
            let expected = sign * 2.25 / 9.16_f64.sqrt();
            assert!((actual - expected).abs() <= 8.0 * f64::EPSILON * expected.abs());
        }
        let error = combined_error(
            0.25,
            scioxide_core::Complex::new(3.0, 4.0),
            scioxide_core::Complex::new(12.0, 0.0),
        )
        .unwrap();
        let correction = 5.0_f64 / 26.44_f64.sqrt();
        assert!((error.real_component() - 0.75 * correction).abs() <= 8.0 * f64::EPSILON);
        assert!((error.imaginary_component() - correction).abs() <= 8.0 * f64::EPSILON);
    }

    #[test]
    fn canonical_tableau_preserves_stage_and_weight_consistency() {
        DOP853_NODES
            .iter()
            .zip(DOP853_STAGE_WEIGHTS)
            .enumerate()
            .for_each(|(stage, (&node, row))| {
                assert_eq!(row.len(), stage);
                assert!(row.iter().all(|coefficient| coefficient.is_finite()));
                assert_consistent_sum(row, node);
            });
        for (offset, (&node, row)) in DENSE_NODES.iter().zip(DENSE_STAGES).enumerate() {
            assert_eq!(row.len(), 13 + offset);
            assert_consistent_sum(row, node);
        }
        for row in DENSE_WEIGHTS {
            assert_consistent_sum(&row, 0.0);
        }
        assert_consistent_sum(&DOP853_WEIGHTS, 1.0);
        assert_consistent_sum(&DOP853_ERROR_WEIGHTS, 0.0);
        assert_consistent_sum(&DOP853_ERROR3_WEIGHTS, 0.0);
    }

    #[test]
    fn stage_combination_keeps_the_step_factor_until_final_rounding() {
        let stages = [Vector::from_vec(alloc::vec![1e308_f64])];
        let initial = Vector::from_vec(alloc::vec![1.0]);
        let mut output = Vector::zeros(1).unwrap();
        crate::safety::combine_stages(
            Some(initial.as_slice()),
            1e-308,
            &[2.0],
            &stages,
            output.as_mut_slice(),
        )
        .unwrap();
        assert!((output.as_slice()[0] - 3.0).abs() <= 2.0 * f64::EPSILON);
    }

    #[test]
    fn component_correction_preserves_e3_and_range() {
        for sign in [-1.0_f64, 1.0] {
            let expected = sign * (1.0 / 1024.0_f64).powi(2) / (1.0 / 1024.0_f64).hypot(0.1);
            let actual = combined_error(1.0, sign / 1024.0, 1.0).unwrap();
            assert!((actual - expected).abs() <= 2.0 * f64::EPSILON * expected.abs());
            let actual32 = combined_error(1.0_f32, sign as f32 / 1024.0, 1.0).unwrap();
            assert!(
                (f64::from(actual32) - expected).abs()
                    <= 2.0 * f64::from(f32::EPSILON) * expected.abs()
            );
        }
        assert_eq!(combined_error(1.0_f64, 0.0, 0.0).unwrap(), 0.0);
        assert_eq!(combined_error(1.0_f64, 0.0, 1.0).unwrap(), 0.0);
        assert_eq!(combined_error(-0.5_f64, 2.0, 0.0).unwrap(), -1.0);
        assert!(combined_error(1.8_f64, 1e308, 1.7e308).unwrap().is_finite());
        assert!(combined_error(1.0_f64, 1e-300, 1e-300).unwrap() > 0.0);
        let tiny = combined_error(1e300_f64, 1e-200, 1e200).unwrap();
        assert!((tiny / 1e-299 - 1.0).abs() < 8.0 * f64::EPSILON);
        let tiny32 = combined_error(1e30_f32, 1e-20_f32, 1e20_f32).unwrap();
        assert!((tiny32 / 1e-29 - 1.0).abs() < 8.0 * f32::EPSILON);
    }
}

#[cfg(test)]
mod order_tests {
    use super::super::order_conditions::{
        alternating, assert_dense, assert_row_sums, assert_weights,
    };

    #[test]
    fn method_estimators_and_continuous_extension_have_their_orders() {
        let tableau = super::tableau::<f64>();
        // Twelve advancing stages, the endpoint derivative and three extension stages.
        assert_row_sums("DOP853", &tableau.c, &tableau.a);
        assert_weights("DOP853", &tableau.a, 8, &tableau.b);
        let fifth = core::array::from_fn(|i| tableau.b[i] - tableau.error[i]);
        assert_weights("DOP853 fifth-order estimator", &tableau.a, 5, &fifth);
        let third = core::array::from_fn(|i| tableau.b[i] - tableau.correction[i]);
        assert_weights("DOP853 third-order estimator", &tableau.a, 3, &third);
        assert_eq!(tableau.c[12], 1.0);
        assert_eq!(tableau.a[12], tableau.b);
        assert_dense("DOP853 dense", &tableau.a, 7, |theta| {
            alternating(&tableau.b, 12, &tableau.dense, theta)
        });
    }
}
