//! Error-estimate contract: accurate values are never downgraded, bounds scale
//! with subnormal values, and tolerances exist only where they steer work.
//! References are mpmath 1.3.0 values at 60–80 digits for exact binary inputs.
#![allow(clippy::unwrap_used)]
use scioxide_core::{Complex, EstimateReliability, Tolerance};
use scioxide_special::*;

/// Half the spacing of binary64 at a reference: the rounding of the reference.
fn half_spacing(reference: f64) -> f64 {
    let magnitude = reference.abs();
    0.5 * (f64::from_bits(magnitude.to_bits() + 1) - magnitude)
}

/// The detailed result is a successful evaluation whose reported error covers
/// the rounded reference.
fn assert_covers(result: &SpecialFunctionResult<f64>, reference_bits: u64) {
    let reference = f64::from_bits(reference_bits);
    assert!(
        result.termination().is_success(),
        "{:?} for reference {reference:e}",
        result.termination()
    );
    let value = result.value().unwrap();
    let observed = (value.value() - reference).abs();
    assert!(
        observed <= value.absolute_error() + half_spacing(reference),
        "{} differs from {reference:e} by {observed:e}, reported {:e}",
        value.value(),
        value.absolute_error()
    );
}

#[test]
fn accurate_values_are_never_downgraded() {
    // Issue #1042: a fixed model once exceeded the default tolerance here.
    let u = -99.87956228626634_f64;
    let m = -0.6996802222550726_f64;
    let sn = jacobi_sn_with_options(u, m, EllipticOptions::default()).unwrap();
    assert_covers(&sn, 0xbfef_af73_57d1_46d6);
    assert_eq!(jacobi_sn(u, m).unwrap(), *sn.value().unwrap().value());
    assert_covers(
        &bessel_j_with_options(2.5, 3.0, BesselOptions::default()).unwrap(),
        0x3fda_69d7_56c4_98f9,
    );
    assert_covers(
        &elliptic_k_with_options(0.5, EllipticOptions::default()).unwrap(),
        0x3ffd_aa4a_3575_9e4b,
    );
    assert_covers(
        &gamma_with_options(3.25, GammaOptions::default()).unwrap(),
        0x4004_64e0_d629_3ffd,
    );
    assert_covers(
        &expint_e1_with_options(1.5, IntegralOptions::default()).unwrap(),
        0x3fb9_9ae2_2365_6477,
    );
    assert_covers(
        &hyp2f1_with_options(0.5, 0.75, 2.5, 0.3, HypergeometricOptions::default()).unwrap(),
        0x3ff0_d0ee_e960_378d,
    );
}

#[test]
fn subnormal_incomplete_gamma_and_beta_return_their_values() {
    // Both values are representable subnormals; fixed absolute floors once
    // overflowed the scaled error coordinate.
    let p = gamma_p_with_options(
        61.52736434399862,
        0.00013922837441179592,
        GammaOptions::default(),
    )
    .unwrap();
    assert_covers(&p, 0x19);
    assert!(p.scaled().is_some());
    assert_eq!(
        gamma_p(61.52736434399862, 0.00013922837441179592).unwrap(),
        f64::from_bits(0x19)
    );
    let q = gamma_q_with_options(
        3.2785691259222554,
        758.3876613191785,
        GammaOptions::default(),
    )
    .unwrap();
    assert_covers(&q, 0x1);
    assert!(q.scaled().is_some());
    let x = 1e-160_f64;
    assert_covers(
        &beta_inc_with_options(2.0, 1.0, x, BetaOptions::default()).unwrap(),
        0x7e8,
    );
    assert_covers(
        &gamma_lower_with_options(2.0, x, GammaOptions::default()).unwrap(),
        0x3f4,
    );
    let log = beta_inc_log_with_options(2.0, 1.0, x, BetaOptions::default()).unwrap();
    assert_covers(&log, 0xc087_069e_2aa2_aa5b);
}

#[test]
fn subnormal_arguments_keep_scale_relative_estimates() {
    let tiny = f64::from_bits(1);
    let sn = jacobi_sn_with_options(tiny, 0.5, EllipticOptions::default()).unwrap();
    assert_eq!(*sn.value().unwrap().value(), tiny);
    assert!(sn.value().unwrap().absolute_error() <= tiny);
    assert!(sn.scaled().is_some());
    let legendre = legendre_p_with_options(1, tiny, LegendreOptions::default()).unwrap();
    assert_eq!(*legendre.value().unwrap().value(), tiny);
    assert!(legendre.scaled().is_some());
    let inverse = erfinv_with_options(tiny, ErrorFunctionOptions::default()).unwrap();
    assert!(inverse.scaled().is_some());
    assert_eq!(
        inverse.value().unwrap().reliability(),
        EstimateReliability::StatisticallyCalibrated
    );
}

/// u, m and mpmath sn, cn, dn for seeded points with m in [-10, 1).
const JACOBI_REFERENCES: [[u64; 5]; 24] = [
    [
        0x4057848f460eb6be,
        0x3feacd8ee87d2140,
        0xbf99107a2713fc31,
        0x3feffd8baefc68a0,
        0x3feffdf1bd576a14,
    ],
    [
        0x4046252d595532bc,
        0xc023a156c3ceab0e,
        0xbfd08f713a7ff03e,
        0x3feee9013fe3944e,
        0x3ff498d0a5c8637d,
    ],
    [
        0xc036ae74b3ef19b0,
        0xc016cb7f75887834,
        0xbfe2ec376434aba0,
        0x3fe9ce3deba21246,
        0x3ffbaddc57a21dc1,
    ],
    [
        0xc043eb62eb1980a8,
        0xc023038a385a065f,
        0xbfd48973bc5498f1,
        0xbfee4ec79e6476d3,
        0x3ff6820563ba0939,
    ],
    [
        0xc057321adca4fa7b,
        0xc002940490e23210,
        0xbfe140161b17ded6,
        0xbfeaf3c93026a64e,
        0x3ff4b4e203f7ac12,
    ],
    [
        0x404476918ed421d8,
        0xc01ccf15a4da2553,
        0xbfed87134f185f90,
        0x3fd8aad24f2d3124,
        0x40055d784b6bc648,
    ],
    [
        0xc043ff6c8b2f928c,
        0xc0053385d5995312,
        0xbfc0ce10b0a571f5,
        0x3fefb91772c1824c,
        0x3ff05c820806a5f5,
    ],
    [
        0x403b61ab8515c220,
        0xc010aa9aa619acdf,
        0xbfe607b538592f8d,
        0x3fe735987cb6be70,
        0x3ffb98949d0b63fb,
    ],
    [
        0xc0352dc2eee87300,
        0xc0111336eca9882f,
        0xbfe91e7f97c587a0,
        0xbfe3d330a827306c,
        0x3ffe7c4a9745ec52,
    ],
    [
        0x40152aa09061d630,
        0x3fec0e2ae887e390,
        0xbfd48334befe7d85,
        0xbfee4fd650423cf2,
        0x3fee8666d72d7858,
    ],
    [
        0x4044569ad8b6a09c,
        0xc0235e48589ebc60,
        0xbfefca3e344db193,
        0xbfbd478e75329f80,
        0x4009fe69dc1c1168,
    ],
    [
        0xc047ed8f07ad4f4a,
        0xc01b39c381687d45,
        0x3fd25d4518658dd0,
        0xbfeea78201e7cb68,
        0x3ff3fc9490f12bce,
    ],
    [
        0x403909eff80fb04c,
        0xbfcf4d45d402ed00,
        0x3fef014ff8a558b8,
        0x3fcfab37b386569a,
        0x3ff1bde6b9b06023,
    ],
    [
        0xc0510e5d9dd99ca3,
        0xbfe15765ac5759c0,
        0xbfe6dbaaf442c631,
        0x3fe664f627d52a39,
        0x3ff213c6b35f9675,
    ],
    [
        0xc04759e16fcf5817,
        0xc019633e67b25cc0,
        0xbf7d66904629c1dd,
        0x3fefffc9f975f20b,
        0x3ff000ab6e7ea658,
    ],
    [
        0x404bad118f37c1c4,
        0xc01aa03f7e379e6b,
        0xbfdbc6504cbd2027,
        0xbfecd45c560161fc,
        0x3ff8050211e3824e,
    ],
    [
        0x405154ac47c12f62,
        0xbffffe391e890b90,
        0xbfee08353f729dfd,
        0x3fd618f7cf4f4d92,
        0x3ffa9640b51b8152,
    ],
    [
        0xc056773fec35e404,
        0xc0154948ea797354,
        0x3fddc03bd66cd38f,
        0x3fec55136db08653,
        0x3ff775d986e01ef5,
    ],
    [
        0xc042d329082fb1d2,
        0xc014b9318b4a9b7e,
        0x3fce07fc94ff428f,
        0x3fef1b572ec2914e,
        0x3ff22378a8fc2321,
    ],
    [
        0x4044f44e756f81c4,
        0xc010e772a857765b,
        0xbfb1b46feafce87c,
        0xbfefec629c5d3a67,
        0x3ff0293088828e14,
    ],
    [
        0x404d8dc7ea9f81c4,
        0xc018d1ddd49026dc,
        0x3fe3acaf07d11d3c,
        0xbfe93cb3ed431dff,
        0x3ffd43f2940acb35,
    ],
    [
        0xc0558ca6f0d8f213,
        0xc015bd407f14b172,
        0xbfc2f6d29cbeddff,
        0x3fefa597633cc50b,
        0x3ff0ed7082655938,
    ],
    [
        0xbff535bcea260540,
        0xbfe6624b7e6fdfe0,
        0xbfeff27752fe11ba,
        0x3fad6b28bd2e34b7,
        0x3ff4d81d401f1844,
    ],
    [
        0xc056217aec2dfad6,
        0xc0124c76721db2a9,
        0x3fe5c02da3985df5,
        0xbfe778ae7de18756,
        0x3ffc3b75e02114cd,
    ],
];

#[test]
fn jacobi_errors_cover_independent_references() {
    let options = EllipticOptions::default();
    for row in JACOBI_REFERENCES {
        let (u, m) = (f64::from_bits(row[0]), f64::from_bits(row[1]));
        assert_covers(&jacobi_sn_with_options(u, m, options).unwrap(), row[2]);
        assert_covers(&jacobi_cn_with_options(u, m, options).unwrap(), row[3]);
        assert_covers(&jacobi_dn_with_options(u, m, options).unwrap(), row[4]);
    }
}

/// Deterministic splitmix64 samples in [0, 1).
fn uniform(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    (z ^ (z >> 31)) as f64 * 2f64.powi(-64)
}

#[test]
fn jacobi_grid_returns_values_with_consistent_errors() {
    let options = EllipticOptions::default();
    let mut state = 1042;
    for _ in 0..4000 {
        let u = -100.0 + 200.0 * uniform(&mut state);
        let m = -10.0 + 11.0 * uniform(&mut state);
        let [sn, cn, dn] = [
            jacobi_sn_with_options(u, m, options).unwrap(),
            jacobi_cn_with_options(u, m, options).unwrap(),
            jacobi_dn_with_options(u, m, options).unwrap(),
        ]
        .map(|result| {
            assert!(result.termination().is_success(), "u={u}, m={m}");
            let value = result.value().unwrap();
            (*value.value(), value.absolute_error())
        });
        assert_eq!(jacobi_sn(u, m).unwrap(), sn.0);
        // sn² + cn² = 1 and dn² + m sn² = 1 within the reported errors.
        let pythagorean = (sn.0 * sn.0 + cn.0 * cn.0 - 1.0).abs();
        assert!(
            pythagorean
                <= 2.0 * (sn.0.abs() * sn.1 + cn.0.abs() * cn.1)
                    + sn.1 * sn.1
                    + cn.1 * cn.1
                    + 4.0 * f64::EPSILON,
            "u={u}, m={m}: {pythagorean:e}"
        );
        let modular = (dn.0 * dn.0 + m * sn.0 * sn.0 - 1.0).abs();
        assert!(
            modular
                <= 2.0 * (dn.0.abs() * dn.1 + m.abs() * sn.0.abs() * sn.1)
                    + dn.1 * dn.1
                    + m.abs() * sn.1 * sn.1
                    + 4.0 * f64::EPSILON * (1.0 + m.abs() + dn.0 * dn.0),
            "u={u}, m={m}: {modular:e}"
        );
    }
    let mut state = 2042;
    for _ in 0..500 {
        let u = (-100.0 + 200.0 * uniform(&mut state)) as f32;
        let m = (-10.0 + 11.0 * uniform(&mut state)) as f32;
        let result = jacobi_sn_with_options(u, m, options).unwrap();
        assert!(result.termination().is_success(), "u={u}, m={m}");
        let reference = jacobi_sn(f64::from(u), f64::from(m)).unwrap();
        let value = result.value().unwrap();
        assert!(
            (f64::from(*value.value()) - reference).abs()
                <= f64::from(value.absolute_error()) + 1e-12,
            "u={u}, m={m}"
        );
    }
}

#[test]
fn spectral_tolerance_steers_truncation() {
    let tight = MathieuPlan::new(
        0,
        MathieuParity::Even,
        50.0_f64,
        scioxide_special::SpectralNormalization::Standard,
        SpectralOptions::default(),
    )
    .unwrap();
    let loose_options =
        SpectralOptions::default().with_tolerance(Tolerance::new(1e-6, 1e-6).unwrap());
    let loose = MathieuPlan::new(
        0,
        MathieuParity::Even,
        50.0_f64,
        scioxide_special::SpectralNormalization::Standard,
        loose_options,
    )
    .unwrap();
    assert!(loose.diagnostics().work() < tight.diagnostics().work());
    assert!(loose.diagnostics().terms() < tight.diagnostics().terms());
    let characteristic = tight.characteristic_value().unwrap();
    assert!(
        (loose.characteristic_value().unwrap() - characteristic).abs()
            <= 1e-6 * characteristic.abs()
    );
}

#[test]
fn ellipsoidal_tolerance_steers_quadrature() {
    let options = EllipsoidalOptions::<f64>::default();
    let loose =
        options.with_relative_tolerance(scioxide_core::RelativeTolerance::new(1e-6).unwrap());
    let plan = EllipsoidalHarmonicPlan::new(0.5, 1.0, 6, 1, options).unwrap();
    let s = Complex::new(1.05, 0.1);
    let tight = plan.second_kind(s).unwrap();
    let coarse = EllipsoidalHarmonicPlan::new(0.5, 1.0, 6, 1, loose)
        .unwrap()
        .second_kind(s)
        .unwrap();
    assert!(coarse.counts().iterations() < tight.counts().iterations());
    let (tight, coarse) = (tight.value().unwrap(), coarse.value().unwrap());
    assert!(
        (*tight.value() - *coarse.value()).norm()
            <= tight.absolute_error() + coarse.absolute_error()
    );
}

#[test]
fn reliability_classes_follow_the_estimate_basis() {
    // A derived series bound, an oracle-qualified envelope and an exact value.
    let derived = gamma_p_with_options(2.5, 0.75, GammaOptions::default()).unwrap();
    assert_eq!(
        derived.value().unwrap().reliability(),
        EstimateReliability::ModelBased
    );
    let calibrated = bessel_j_with_options(0.5, 2.0, BesselOptions::default()).unwrap();
    assert_eq!(
        calibrated.value().unwrap().reliability(),
        EstimateReliability::StatisticallyCalibrated
    );
    let exact = gamma_with_options(5.0, GammaOptions::default()).unwrap();
    assert_eq!(exact.termination(), SpecialFunctionTermination::Exact);
    assert_eq!(exact.value().unwrap().absolute_error(), 0.0);
    assert_eq!(exact.try_value().unwrap(), 24.0);
}
