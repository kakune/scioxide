//! Error function family: erf, erfc, erfinv, erfcinv.
//!
//! # Algorithms
//!
//! - **erf / erfc** use fixed-cost kernels for `f32` and `f64`: a direct
//!   central erfc polynomial and an erf rational approximation below one, and
//!   `exp(-x²)` times the crate's single erfcx kernel beyond.
//! - **erfinv / erfcinv** use a scalar-native fixed-cost, piecewise rational
//!   approximation. `erfcinv` forms its tail logarithm directly from the
//!   complement, so representable tail probabilities are never rounded away.
//!   The final rational range covers every positive `f64` subnormal.

use crate::support::polynomial::{PolynomialPolicy, polynomial};
use crate::support::{check_finite, require_finite_param};
use crate::util::{is_exact, is_exact_zero};
use scioxide_core::{RealScalar, SciOxideError};

#[derive(Clone, Copy)]
enum ErrorFunction {
    Erf,
    Erfc,
}

impl ErrorFunction {
    #[inline]
    fn primitive_f64(self, x: f64) -> f64 {
        match self {
            Self::Erf => erf_f64(x),
            Self::Erfc => erfc_f64(x),
        }
    }
}

#[inline]
fn checked_error_function<S: RealScalar>(
    x: S,
    function: ErrorFunction,
) -> Result<S, SciOxideError> {
    require_finite_param(x, "probability parameter x must be finite")?;

    let primitive = check_finite(function.primitive_f64(x.try_to_f64_exact()?))?;
    let value = S::try_from_f64_lossy(primitive)?;

    check_finite(value)
}

/// erf.
#[inline]
pub(crate) fn erf<S: RealScalar>(x: S) -> Result<S, SciOxideError> {
    checked_error_function(x, ErrorFunction::Erf)
}

/// erfc.
#[inline]
pub(crate) fn erfc<S: RealScalar>(x: S) -> Result<S, SciOxideError> {
    checked_error_function(x, ErrorFunction::Erfc)
}

// Real implementation of the public crate::erfinv contract.
#[inline]
pub(crate) fn erfinv<S: RealScalar>(x: S) -> Result<S, SciOxideError> {
    if let Some(result) = crate::support::delegate_f32::<S, _, _>(
        || erfinv::<f64>(x.into_f64()),
        |value| {
            if value == f64::INFINITY {
                Ok(S::INFINITY)
            } else if value == f64::NEG_INFINITY {
                Ok(S::NEG_INFINITY)
            } else {
                S::try_from_f64_lossy(value)
            }
        },
    ) {
        return result;
    }
    require_finite_param(x, "probability parameter x must be finite")?;
    if is_exact(x, -S::ONE) {
        return Ok(S::NEG_INFINITY);
    }
    if is_exact(x, S::ONE) {
        return Ok(S::INFINITY);
    }
    if x < -S::ONE || x > S::ONE {
        return Err(SciOxideError::domain_error(
            scioxide_core::argument!("x"),
            scioxide_core::DomainReason::OutsideDomain,
            "inverse erf argument must be in [-1, 1]",
        ));
    }
    if is_exact_zero(x) {
        return Ok(x);
    }

    let magnitude = x.abs();
    inverse_error_rational(x, S::ONE - magnitude)
}

// Real implementation of the public crate::erfcinv contract.
#[inline]
pub(crate) fn erfcinv<S: RealScalar>(x: S) -> Result<S, SciOxideError> {
    if let Some(result) = crate::support::delegate_f32::<S, _, _>(
        || erfcinv::<f64>(x.into_f64()),
        |value| {
            if value == f64::INFINITY {
                Ok(S::INFINITY)
            } else if value == f64::NEG_INFINITY {
                Ok(S::NEG_INFINITY)
            } else {
                S::try_from_f64_lossy(value)
            }
        },
    ) {
        return result;
    }
    require_finite_param(x, "probability parameter x must be finite")?;
    let two = S::TWO;
    if is_exact_zero(x) {
        return Ok(S::INFINITY);
    }
    if is_exact(x, two) {
        return Ok(S::NEG_INFINITY);
    }
    if x < S::ZERO || x > two {
        return Err(SciOxideError::domain_error(
            scioxide_core::argument!("x"),
            scioxide_core::DomainReason::OutsideDomain,
            "inverse erfc argument must be in [0, 2]",
        ));
    }
    let signed_probability = S::ONE - x;
    let complement = if x <= S::ONE { x } else { two - x };
    inverse_error_rational(signed_probability, complement)
}

// Direct degree-18 Chebyshev interpolant of erfc(0.5 + t), |t| <= 0.5. Generated independently
// at 160/240 digits by generate_erfc.py. The analytic approximation bound
// is below 3e-18.
const ERFC_CENTRAL: [f64; 19] = [
    f64::from_bits(0x3fdeb02147ce245c),
    f64::from_bits(0xbfec1efca49a5012),
    f64::from_bits(0x3fdc1efca49a5011),
    f64::from_bits(0x3fc2bf531866e141),
    f64::from_bits(0xbfc76f27de80980b),
    f64::from_bits(0xbf7dfeeb5a48d584),
    f64::from_bits(0x3fa99f13b26a7285),
    f64::from_bits(0xbf7623c6169c24ea),
    f64::from_bits(0xbf8493d4809116e6),
    f64::from_bits(0x3f61c16430a29754),
    f64::from_bits(0x3f59b6f2532eb7c7),
    f64::from_bits(0xbf404c094749a83d),
    f64::from_bits(0xbf29bca3b9537c22),
    f64::from_bits(0x3f16572736876732),
    f64::from_bits(0x3ef4c423b04b9a99),
    f64::from_bits(0xbee8bcafde5473aa),
    f64::from_bits(0xbeba4c93238410ec),
    f64::from_bits(0x3eb4d3274d0d814f),
    f64::from_bits(0x3e76b5c5287100dc),
];

// Cephes-derived erf rational for |x| <= 1. Coefficients are evaluated in f64;
// f32 inputs are widened exactly and narrowed only after the bounded result is
// known. The formula is fixed-cost and contains no fallible operations.

const ERF_T: [f64; 5] = [
    9.604_973_739_870_516,
    90.026_019_720_384_27,
    2_232.005_345_946_843,
    7_003.325_141_128_051,
    55_592.301_301_039_5,
];
const ERF_U: [f64; 5] = [
    33.561_714_164_750_31,
    521.357_949_780_152_7,
    4_594.323_829_709_801,
    22_629.000_061_389_094,
    49_267.394_260_863_59,
];
#[inline]
fn erf_f64(x: f64) -> f64 {
    if x == 0.0 {
        return x;
    }
    if x.abs() > 1.0 {
        return if x < 0.0 {
            erfc_f64(-x) - 1.0
        } else {
            1.0 - erfc_f64(x)
        };
    }
    let square = x * x;
    x * polynomial(square, &ERF_T, PolynomialPolicy::UnfusedDescending)
        / polynomial(square, &ERF_U, PolynomialPolicy::MonicDescending)
}

#[inline(always)]
fn erfc_f64(x: f64) -> f64 {
    let absolute = x.abs();
    if absolute == 0.0 {
        return 1.0;
    }
    if absolute < 1.0 {
        let value = polynomial(absolute - 0.5, &ERFC_CENTRAL, PolynomialPolicy::Grouped);
        return if x < 0.0 { 2.0 - value } else { value };
    }
    if absolute > 27.3 {
        return if x < 0.0 { 2.0 } else { 0.0 };
    }

    let square = absolute * absolute;
    let square_error = crate::support::wide::two_product(absolute, absolute).lo;
    let base = (-square).exp();
    // Correct the argument rounding in exp(-x*x). For |x| <= 27.3 the
    // omitted quadratic term in exp(-square_error) is far below one ULP.
    // Normal correction-product rounding contributes at most
    // epsilon*|square_error| relative error. Subnormal corrections retain the
    // same strict tail ULP gate. Ordinary subtraction keeps the compensation
    // without a second portable FMA call.
    let exponential = base - base * square_error;
    let value = exponential * crate::error_probability::real_faddeeva::erfcx(absolute);
    if x < 0.0 { 2.0 - value } else { value }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

// The following 64-bit coefficient tables and region boundaries are adapted
// from Boost.Math's inverse-erf implementation:
// https://github.com/boostorg/math/blob/develop/include/boost/math/special_functions/detail/erf_inv.hpp
// Copyright John Maddock 2006 and Matt Borland 2024; distributed under the
// Boost Software License, Version 1.0. Only regions reachable by finite real
// complementary probabilities are retained (sqrt(-ln(q)) < 28 for binary64).
#[allow(clippy::excessive_precision)]
mod erfinv_coefficients {
    pub(super) const CENTRAL_P: [f64; 8] = [
        -0.000_508_781_949_658_280_7,
        -0.008_368_748_197_417_368,
        0.033_480_662_540_974_46,
        -0.012_692_614_766_297_403,
        -0.036_563_797_141_176_27,
        0.021_987_868_111_116_89,
        0.008_226_878_746_769_157,
        -0.005_387_729_650_712_429,
    ];
    pub(super) const CENTRAL_Q: [f64; 10] = [
        1.0,
        -0.970_005_043_303_290_6,
        -1.565_745_582_341_758_4,
        1.562_215_583_984_230_2,
        0.662_328_840_472_003,
        -0.712_289_023_415_428_5,
        -0.052_739_638_234_009_97,
        0.079_528_368_734_157_17,
        -0.002_333_937_593_741_9,
        0.000_886_216_390_456_424_7,
    ];
    pub(super) const SHOULDER_P: [f64; 9] = [
        -0.202_433_508_355_938_76,
        0.105_264_680_699_391_71,
        8.370_503_283_431_199,
        17.644_729_840_837_4,
        -18.851_064_805_871_425,
        -44.638_232_444_178_7,
        17.445_385_985_570_866,
        21.129_465_544_834_05,
        -3.671_922_547_077_293_6,
    ];
    pub(super) const SHOULDER_Q: [f64; 9] = [
        1.0,
        6.242_641_248_542_475,
        3.971_343_795_334_386,
        -28.660_818_049_98,
        -20.143_263_468_048_52,
        48.560_921_310_873_99,
        10.826_866_735_546_016,
        -22.643_693_341_313_972,
        1.721_147_657_612_002_8,
    ];
    pub(super) const TAIL_NEAR_P: [f64; 11] = [
        -0.131_102_781_679_951_9,
        -0.163_794_047_193_317_06,
        0.117_030_156_341_995_25,
        0.387_079_738_972_604_34,
        0.337_785_538_912_035_9,
        0.142_869_534_408_157_15,
        0.029_015_791_000_532_906,
        0.002_145_589_953_888_052_8,
        -0.000_000_679_465_575_181_126_3,
        0.000_000_028_522_533_178_221_706,
        -0.000_000_000_681_149_956_853_777,
    ];
    pub(super) const TAIL_NEAR_Q: [f64; 8] = [
        1.0,
        3.466_254_072_425_672_3,
        5.381_683_457_070_069,
        4.778_465_929_458_438,
        2.593_019_216_236_202_7,
        0.848_854_343_457_902,
        0.152_264_338_295_331_8,
        0.011_059_242_293_464_891,
    ];
    pub(super) const TAIL_MIDDLE_P: [f64; 9] = [
        -0.035_035_378_718_317_8,
        -0.002_224_265_292_134_479_3,
        0.018_557_330_651_423_108,
        0.009_508_047_013_259_197,
        0.001_871_234_928_195_592_3,
        0.000_157_544_617_424_960_55,
        0.000_004_604_698_905_843_18,
        -0.000_000_000_230_404_776_911_882_6,
        0.000_000_000_002_663_392_274_257_820_4,
    ];
    pub(super) const TAIL_MIDDLE_Q: [f64; 7] = [
        1.0,
        1.365_334_981_755_406_4,
        0.762_059_164_553_623_4,
        0.220_091_105_764_131_25,
        0.034_158_914_367_094_774,
        0.002_638_616_766_570_16,
        0.000_076_467_529_230_279_45,
    ];
    pub(super) const TAIL_FAR_P: [f64; 9] = [
        -0.016_743_100_507_663_374,
        -0.001_129_514_387_455_802_8,
        0.001_056_288_621_524_929,
        0.000_209_386_317_487_588_08,
        0.000_014_962_478_375_834_237,
        0.000_000_449_696_789_927_706_43,
        0.000_000_004_625_961_635_228_786,
        -0.000_000_000_000_028_112_873_562_883_18,
        0.000_000_000_000_000_099_055_709_973_310_32,
    ];
    pub(super) const TAIL_FAR_Q: [f64; 7] = [
        1.0,
        0.591_429_344_886_417_5,
        0.138_151_865_749_083_33,
        0.016_074_608_709_367_65,
        0.000_964_011_807_005_165_5,
        0.000_027_533_547_476_472_605,
        0.000_000_282_243_172_016_108,
    ];
    pub(super) const TAIL_EXTREME_P: [f64; 8] = [
        -0.0024978212791898131227,
        -0.779190719229053954292e-5,
        0.254723037413027451751e-4,
        0.162397777342510920873e-5,
        0.396341011304801168516e-7,
        0.411632831190944208473e-9,
        0.145596286718675035587e-11,
        -0.116765012397184275695e-17,
    ];
    pub(super) const TAIL_EXTREME_Q: [f64; 7] = [
        1.0,
        0.207123112214422517181,
        0.0169410838120975906478,
        0.000690538265622684595676,
        0.145007359818232637924e-4,
        0.144437756628144157666e-6,
        0.509761276599778486139e-9,
    ];
}

// Keep table values visible as well as lengths so equally sized tails
// can specialize independently through the shared polynomial evaluator.
#[inline(always)]
fn rational<S: RealScalar, const N: usize, const M: usize>(
    x: S,
    numerator: &[f64; N],
    denominator: &[f64; M],
) -> Result<S, SciOxideError> {
    // Every containing evaluator delegates binary32 before reaching this
    // kernel. Keep binary64 tables visible rather than materializing arrays.
    let x = x.try_to_f64_exact()?;
    S::try_from_f64_lossy(
        polynomial(x, numerator, PolynomialPolicy::Fused)
            / polynomial(x, denominator, PolynomialPolicy::Fused),
    )
}

/// Inverse erf from a signed erf probability and its positive complementary
/// magnitude. Callers validate the domain and handle endpoint limits first.
pub(super) fn inverse_error_rational<S: RealScalar>(
    signed_probability: S,
    complement: S,
) -> Result<S, SciOxideError> {
    use erfinv_coefficients as coefficients;

    let p = signed_probability.abs();
    if p <= S::HALF {
        let ten = S::TEN;
        let gain = signed_probability * (p + ten);
        let correction = rational(p, &coefficients::CENTRAL_P, &coefficients::CENTRAL_Q)?;
        let constant = S::try_from_f64_lossy(0.089_131_474_494_934_08)?;
        return check_finite(scioxide_core::fast_mul_add(
            gain,
            constant,
            gain * correction,
        ));
    }

    let negative_log_q = -complement.ln();
    let magnitude = if complement >= S::try_from_f64_lossy(0.25)? {
        let correction = rational(
            complement - S::try_from_f64_lossy(0.25)?,
            &coefficients::SHOULDER_P,
            &coefficients::SHOULDER_Q,
        )?;
        ((S::TWO) * negative_log_q).sqrt()
            / (S::try_from_f64_lossy(2.249_481_201_171_875)? + correction)
    } else {
        let tail = negative_log_q.sqrt();
        // Evaluate each fixed-size table in its own branch. Selecting slices
        // first loses the lengths needed to unroll the fused Horner chains.
        let (constant, correction) = if tail < S::try_from_f64_lossy(3.0)? {
            (
                0.807_220_458_984_375,
                rational(
                    tail - S::try_from_f64_lossy(1.125)?,
                    &coefficients::TAIL_NEAR_P,
                    &coefficients::TAIL_NEAR_Q,
                )?,
            )
        } else if tail < S::try_from_f64_lossy(6.0)? {
            (
                0.939_955_711_364_746_1,
                rational(
                    tail - S::try_from_f64_lossy(3.0)?,
                    &coefficients::TAIL_MIDDLE_P,
                    &coefficients::TAIL_MIDDLE_Q,
                )?,
            )
        } else if tail < S::try_from_f64_lossy(18.0)? {
            (
                0.983_628_273_010_253_9,
                rational(
                    tail - S::try_from_f64_lossy(6.0)?,
                    &coefficients::TAIL_FAR_P,
                    &coefficients::TAIL_FAR_Q,
                )?,
            )
        } else {
            (
                0.997_145_652_770_996_1,
                rational(
                    tail - S::try_from_f64_lossy(18.0)?,
                    &coefficients::TAIL_EXTREME_P,
                    &coefficients::TAIL_EXTREME_Q,
                )?,
            )
        };
        scioxide_core::fast_mul_add(tail, S::try_from_f64_lossy(constant)?, tail * correction)
    };

    check_finite(if signed_probability < S::ZERO {
        -magnitude
    } else {
        magnitude
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    // ---- erf ----

    #[test]
    fn test_erf_zero_f64() {
        let val = erf(0.0_f64).unwrap();
        assert!(val.abs() < 1e-15);
    }

    #[test]
    fn test_checked_erf_preserves_signed_zero() {
        assert_eq!(erf(0.0_f64).unwrap().to_bits(), 0.0_f64.to_bits());
        assert_eq!(erf(-0.0_f64).unwrap().to_bits(), (-0.0_f64).to_bits());
        assert_eq!(erf(0.0_f32).unwrap().to_bits(), 0.0_f32.to_bits());
        assert_eq!(erf(-0.0_f32).unwrap().to_bits(), (-0.0_f32).to_bits());
        assert_eq!(erfc(0.0_f64).unwrap(), 1.0);
        assert_eq!(erfc(-0.0_f64).unwrap(), 1.0);
        assert_eq!(erfc(0.0_f32).unwrap(), 1.0);
        assert_eq!(erfc(-0.0_f32).unwrap(), 1.0);
    }

    #[test]
    fn test_erf_symmetry_f64() {
        let x = 0.75_f64;
        let pos = erf(x).unwrap();
        let neg = erf(-x).unwrap();
        assert!((pos + neg).abs() < 1e-14);
    }

    #[test]
    fn test_erf_known_value_f64() {
        // erf(1) ≈ 0.8427007929497149
        let val = erf(1.0_f64).unwrap();
        assert!((val - 0.842_700_792_949_714_9).abs() < 1e-12);
    }

    #[test]
    fn test_erf_large_f64() {
        let val = erf(5.0_f64).unwrap();
        assert!((val - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_erf_huge_finite_input_limits() {
        let pos = erf(f64::MAX).unwrap();
        let neg = erf(-f64::MAX).unwrap();
        assert!((pos - 1.0).abs() < 1e-15);
        assert!((neg + 1.0).abs() < 1e-15);
    }

    #[test]
    fn test_erf_half_f64() {
        // erf(0.5) ≈ 0.5204998778130465
        let val = erf(0.5_f64).unwrap();
        assert!((val - 0.520_499_877_813_046_5).abs() < 1e-12);
    }

    #[test]
    fn test_erf_f32() {
        let val = erf(1.0_f32).unwrap();
        assert!((val - 0.842_700_8_f32).abs() < 1e-5);
    }

    #[test]
    fn test_erf_negative_f32() {
        let val = erf(-0.5_f32).unwrap();
        assert!(val < 0.0);
        assert!((val + 0.520_499_9_f32).abs() < 1e-5);
    }

    // ---- erfc ----

    #[test]
    fn test_erfc_zero_f64() {
        let val = erfc(0.0_f64).unwrap();
        assert!((val - 1.0).abs() < 1e-15);
    }

    #[test]
    fn test_erfc_complement_f64() {
        let x = 0.75_f64;
        let sum = erf(x).unwrap() + erfc(x).unwrap();
        assert!((sum - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_erfc_large_f64() {
        let val = erfc(5.0_f64).unwrap();
        assert!(val > 0.0);
        assert!(val < 1e-10);
    }

    #[test]
    fn test_erfc_huge_finite_input_limits() {
        let pos = erfc(f64::MAX).unwrap();
        let neg = erfc(-f64::MAX).unwrap();
        assert!(pos.abs() < 1e-15);
        assert!((neg - 2.0).abs() < 1e-15);
    }

    #[test]
    fn test_erfc_negative_f64() {
        let val = erfc(-1.0_f64).unwrap();
        assert!((val - (1.0 + 0.842_700_792_949_714_9)).abs() < 1e-12);
    }

    #[test]
    fn test_erfc_f32() {
        let val = erfc(0.0_f32).unwrap();
        assert!((val - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_erfc_central_regressions() {
        [
            (0x3feb_b9b0_b60e_d930, 0x3fcc_381e_087b_8e40),
            (0x3fee_e22e_6c2d_a4bc, 0x3fc6_0db5_a443_0ac8),
        ]
        .into_iter()
        .for_each(|(input, expected)| {
            let x = f64::from_bits(input);
            let value = erfc(x).unwrap();
            assert!(value.to_bits().abs_diff(expected) <= 4);
            assert!((erfc(-x).unwrap() + value - 2.0).abs() <= f64::EPSILON);
        });
    }

    #[test]
    fn test_erfc_monotonic_central_and_cutovers() {
        let mut previous = erfc(-1.0_f64).unwrap();
        for index in 1..=8192 {
            let x = f64::from(index) / 4096.0 - 1.0;
            let value = erfc(x).unwrap();
            assert!(value < previous, "erfc must decrease at {x}");
            previous = value;
        }
        [-27.3_f64, -8.0, -1.0, -0.0, 0.0, 1.0, 8.0, 27.3]
            .into_iter()
            .for_each(|x| {
                let below = erfc(x.next_down()).unwrap();
                let value = erfc(x).unwrap();
                let above = erfc(x.next_up()).unwrap();
                assert!(below >= value && value >= above, "erfc cutover at {x}");
            });
    }

    #[test]
    fn test_error_function_tail_oracles_and_identities() {
        let cases = [
            (1.0_f64, 0.842_700_792_949_714_9, 0.157_299_207_050_285_13),
            (5.0, 0.999_999_999_998_462_6, 1.537_459_794_428_035e-12),
            (10.0, 1.0, 2.088_487_583_762_545e-45),
            (27.0, 1.0, 5.237_05e-319),
        ];

        cases
            .into_iter()
            .for_each(|(x, expected_erf, expected_erfc)| {
                let actual_erf = erf(x).unwrap();
                let actual_erfc = erfc(x).unwrap();
                assert!((actual_erf - expected_erf).abs() <= 2e-15);
                let relative_error = ((actual_erfc - expected_erfc) / expected_erfc).abs();
                assert!(
                    relative_error <= 2e-6,
                    "erfc({x}) relative error {relative_error}"
                );
                assert!((erf(-x).unwrap() + actual_erf).abs() <= 2e-15);
                assert!((erfc(-x).unwrap() + actual_erfc - 2.0).abs() <= 2e-15);
            });
    }

    #[test]
    fn test_checked_non_finite_policy() {
        {
            let outcome = &(erf(f64::NAN));
            let err = outcome.as_ref().expect_err("expected error");
            assert!(matches!(
                err.view(),
                scioxide_core::SciOxideErrorView::NonFinite {
                    origin: scioxide_core::NonFiniteOrigin::Input,
                    ..
                }
            ));
        }
        {
            let outcome = &(erfc(f64::INFINITY));
            let err = outcome.as_ref().expect_err("expected error");
            assert!(matches!(
                err.view(),
                scioxide_core::SciOxideErrorView::NonFinite {
                    origin: scioxide_core::NonFiniteOrigin::Input,
                    ..
                }
            ));
        }

        [f32::NAN, f32::INFINITY, f32::NEG_INFINITY]
            .into_iter()
            .for_each(|x| {
                let erf_error = erf(x).expect_err("expected non-finite error");
                assert!(matches!(
                    erf_error.view(),
                    scioxide_core::SciOxideErrorView::NonFinite { .. }
                ));
                let erfc_error = erfc(x).expect_err("expected non-finite error");
                assert!(matches!(
                    erfc_error.view(),
                    scioxide_core::SciOxideErrorView::NonFinite { .. }
                ));
            });
    }

    // ---- erfinv ----

    #[test]
    fn test_erfinv_zero_f64() {
        let val = erfinv(0.0_f64).unwrap();
        assert!(val.abs() < 1e-15);
    }

    #[test]
    fn test_erfinv_roundtrip_f64() {
        for &x in &[0.1_f64, 0.3, 0.5, 0.7, 0.9, -0.1, -0.5, -0.9] {
            let t = erfinv(x).unwrap();
            let back = erf(t).unwrap();
            assert!(
                (back - x).abs() < 1e-12,
                "erfinv roundtrip failed for x={x}: erf(erfinv({x}))={back}"
            );
        }
    }

    #[test]
    fn test_erfinv_symmetry_f64() {
        let x = 0.6_f64;
        let pos = erfinv(x).unwrap();
        let neg = erfinv(-x).unwrap();
        assert!((pos + neg).abs() < 1e-14);
    }

    #[test]
    fn test_erfinv_domain_error() {
        assert_eq!(erfinv(1.0_f64).unwrap(), f64::INFINITY);
        assert_eq!(erfinv(-1.0_f64).unwrap(), f64::NEG_INFINITY);
        {
            let outcome = &(erfinv(1.5_f64));
            let err = outcome.as_ref().expect_err("expected error");
            assert!(matches!(
                err.view(),
                scioxide_core::SciOxideErrorView::DomainError { .. }
            ));
        }
    }

    #[test]
    fn test_erfinv_f32() {
        for &x in &[0.1_f32, 0.5, 0.9, -0.3] {
            let t = erfinv(x).unwrap();
            let back = erf(t).unwrap();
            assert!(
                (back - x).abs() < 1e-5,
                "f32 erfinv roundtrip failed for x={x}"
            );
        }
    }

    #[test]
    fn test_erfinv_preserves_signed_zero_and_endpoint_limits() {
        assert_eq!(erfinv(0.0_f64).unwrap().to_bits(), 0.0_f64.to_bits());
        assert_eq!(erfinv(-0.0_f64).unwrap().to_bits(), (-0.0_f64).to_bits());
        assert_eq!(erfinv(0.0_f32).unwrap().to_bits(), 0.0_f32.to_bits());
        assert_eq!(erfinv(-0.0_f32).unwrap().to_bits(), (-0.0_f32).to_bits());
        assert_eq!(erfinv(1.0_f64).unwrap(), f64::INFINITY);
        assert_eq!(erfinv(-1.0_f64).unwrap(), f64::NEG_INFINITY);
        assert_eq!(erfinv(1.0_f32).unwrap(), f32::INFINITY);
        assert_eq!(erfinv(-1.0_f32).unwrap(), f32::NEG_INFINITY);
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn test_erfinv_dense_f64_residual_certificate() {
        const STEPS: u64 = 1 << 20;
        let uniform = (0..STEPS).map(|index| index as f64 / STEPS as f64);
        let logarithmic_tail = (1..=53).map(|exponent| 1.0 - 2.0_f64.powi(-exponent));
        let tail_near = -(-9.0_f64).exp_m1();
        let tail_middle = -(-36.0_f64).exp_m1();
        let boundaries = [
            0.5_f64.next_down(),
            0.5,
            0.5_f64.next_up(),
            0.75_f64.next_down(),
            0.75,
            0.75_f64.next_up(),
            tail_near.next_down(),
            tail_near,
            tail_near.next_up(),
            tail_middle.next_down(),
            tail_middle,
            tail_middle.next_up(),
            1.0_f64.next_down(),
        ];

        let mut previous_inverse = f64::NEG_INFINITY;
        for input in (0..STEPS).map(|index| index as f64 / STEPS as f64) {
            let inverse = erfinv(input).unwrap();
            assert!(inverse >= previous_inverse, "erfinv must be monotone");
            previous_inverse = inverse;
        }

        let (max_absolute_residual, max_residual_ulps) = uniform
            .chain(logarithmic_tail)
            .chain(boundaries)
            .filter(|value| *value < 1.0)
            .fold((0.0_f64, 0_u64), |(max_residual, max_ulps), input| {
                let inverse = erfinv(input).unwrap();
                let negative = erfinv(-input).unwrap();
                assert_eq!(negative.to_bits(), (-inverse).to_bits());
                let roundtrip = erf(inverse).unwrap();
                (
                    max_residual.max((roundtrip - input).abs()),
                    max_ulps.max(roundtrip.to_bits().abs_diff(input.to_bits())),
                )
            });

        assert!(max_absolute_residual <= 4.0 * f64::EPSILON);
        assert!(max_residual_ulps <= 4);
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn test_erfinv_dense_f32_residual_certificate() {
        const STEPS: u32 = 1 << 20;
        let uniform = (0..STEPS).map(|index| index as f32 / STEPS as f32);
        let logarithmic_tail = (1..=24).map(|exponent| 1.0 - 2.0_f32.powi(-exponent));
        let tail_near = -(-9.0_f32).exp_m1();
        let boundaries = [
            0.5_f32.next_down(),
            0.5,
            0.5_f32.next_up(),
            0.75_f32.next_down(),
            0.75,
            0.75_f32.next_up(),
            tail_near.next_down(),
            tail_near,
            tail_near.next_up(),
            1.0_f32.next_down(),
        ];

        let mut previous_inverse = f32::NEG_INFINITY;
        for input in (0..STEPS).map(|index| index as f32 / STEPS as f32) {
            let inverse = erfinv(input).unwrap();
            assert!(inverse >= previous_inverse, "erfinv must be monotone");
            previous_inverse = inverse;
        }

        let (max_absolute_residual, max_residual_ulps) = uniform
            .chain(logarithmic_tail)
            .chain(boundaries)
            .filter(|value| *value < 1.0)
            .fold((0.0_f32, 0_u32), |(max_residual, max_ulps), input| {
                let inverse = erfinv(input).unwrap();
                let negative = erfinv(-input).unwrap();
                assert_eq!(negative.to_bits(), (-inverse).to_bits());
                let roundtrip = erf(inverse).unwrap();
                (
                    max_residual.max((roundtrip - input).abs()),
                    max_ulps.max(roundtrip.to_bits().abs_diff(input.to_bits())),
                )
            });

        assert!(max_absolute_residual <= 4.0 * f32::EPSILON);
        assert!(max_residual_ulps <= 4);
    }

    // ---- erfcinv ----

    #[test]
    fn test_erfcinv_roundtrip_f64() {
        for &x in &[0.1_f64, 0.3, 0.5, 1.0, 1.5, 1.9] {
            let t = erfcinv(x).unwrap();
            let back = erfc(t).unwrap();
            assert!(
                (back - x).abs() < 1e-12,
                "erfcinv roundtrip failed for x={x}: erfc(erfcinv({x}))={back}"
            );
        }
    }

    #[test]
    fn test_erfcinv_preserves_extreme_complement() {
        let probability = 1.0e-20_f64;
        let inverse = erfcinv(probability).unwrap();
        assert!(inverse.is_finite() && inverse > 0.0);
        assert!((inverse - 6.601_580_622_355_143).abs() < 2e-14);
        assert!(((erfc(inverse).unwrap() - probability) / probability).abs() < 2e-13);
    }

    #[test]
    fn test_erfcinv_domain_error() {
        assert_eq!(erfcinv(0.0_f64).unwrap(), f64::INFINITY);
        assert_eq!(erfcinv(2.0_f64).unwrap(), f64::NEG_INFINITY);
        {
            let outcome = &(erfcinv(-0.1_f64));
            let err = outcome.as_ref().expect_err("expected error");
            assert!(matches!(
                err.view(),
                scioxide_core::SciOxideErrorView::DomainError { .. }
            ));
        }
    }

    #[test]
    fn test_erfcinv_one_f64() {
        // erfcinv(1) = erfinv(0) = 0
        let val = erfcinv(1.0_f64).unwrap();
        assert!(val.abs() < 1e-15);
    }

    #[test]
    fn test_erfcinv_f32() {
        for &x in &[0.5_f32, 1.0, 1.5] {
            let t = erfcinv(x).unwrap();
            let back = erfc(t).unwrap();
            assert!(
                (back - x).abs() < 1e-5,
                "f32 erfcinv roundtrip failed for x={x}"
            );
        }
    }
}
