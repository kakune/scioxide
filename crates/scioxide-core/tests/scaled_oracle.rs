//! Current-checkout MPFR/MPC coverage, with independent IEEE output bits.
#![allow(clippy::unwrap_used)]
use scioxide_core::{Complex, LogValue, NumericalIssue, ScaledValue, SciOxideErrorView};

macro_rules! oracle {
    ($name:ident, $s:ty, $bits:ty, $width:literal) => {
        #[test]
        fn $name() {
            let mut cases = 0;
            let mut families = [0; 4];
            for row in include_str!("data/scaled-values.txt")
                .lines()
                .filter(|s| !s.starts_with('#'))
            {
                let fields: Vec<_> = row.split_whitespace().collect();
                if fields[1] != $width {
                    continue;
                }
                cases += 1;
                let decode = |s| <$s>::from_bits(<$bits>::from_str_radix(s, 16).unwrap());
                let close = |actual: $s, expected: $s, ulps: u64| {
                    if expected == 0.0 || !expected.is_finite() {
                        assert_eq!(
                            actual.to_bits(),
                            expected.to_bits(),
                            "{row}: {actual:?} vs {expected:?}"
                        );
                    } else {
                        assert!(
                            actual.is_finite()
                                && u64::from(actual.to_bits().abs_diff(expected.to_bits())) <= ulps,
                            "{row}: {actual:?} vs {expected:?}"
                        );
                    }
                };
                match fields[0] {
                    "component-log" => {
                        families[3] += 1;
                        let value = scioxide_core::Scaled::from_parts(
                            decode(fields[2]),
                            fields[3].parse().unwrap(),
                        )
                        .unwrap();
                        let actual = value.ln_abs().unwrap();
                        let expected = decode(fields[4]);
                        let error = (f64::from(actual) - f64::from(expected)).abs();
                        let bound =
                            16.0 * f64::from(<$s>::EPSILON) * (1.0 + f64::from(expected).abs());
                        assert!(error <= bound, "{row}: {error:e} exceeds {bound:e}");
                    }
                    "exp" => {
                        families[0] += 1;
                        let input = decode(fields[2]);
                        let result = ScaledValue::from_log(1.0 as $s, input);
                        if fields[6] == "0" {
                            assert!(
                                matches!(
                                    result.unwrap_err().view(),
                                    SciOxideErrorView::NumericalInstability {
                                        issue: NumericalIssue::ExponentOutOfRange,
                                        ..
                                    }
                                ),
                                "{row}"
                            );
                            continue;
                        }
                        let actual = result.unwrap();
                        let expected_mantissa = decode(fields[3]);
                        let expected_exponent = fields[4].parse::<i64>().unwrap();
                        let delta =
                            i128::from(actual.real().exponent()) - i128::from(expected_exponent);
                        assert!(delta.abs() <= 1, "{row}: exponent delta {delta}");
                        let aligned =
                            f64::from(actual.real().significand()) * 2.0_f64.powi(delta as i32);
                        let relative =
                            (aligned - f64::from(expected_mantissa)) / f64::from(expected_mantissa);
                        let bound = 16.0 * f64::from(<$s>::EPSILON)
                            + 64.0
                                * (expected_exponent as f64).abs()
                                * (f64::EPSILON / 2.0).powi(2);
                        assert!(
                            relative.ln_1p().abs() <= bound,
                            "{row}: log error {relative:e}, bound {bound:e}"
                        );
                        let ordinary = decode(fields[5]);
                        if ordinary.is_infinite() {
                            assert!(
                                matches!(
                                    actual.round().unwrap_err().view(),
                                    SciOxideErrorView::NumericalInstability {
                                        issue: NumericalIssue::Overflow,
                                        ..
                                    }
                                ),
                                "{row}"
                            );
                        } else {
                            close(actual.round().unwrap(), ordinary, 4);
                        }
                    }
                    "log" => {
                        families[1] += 1;
                        let input = Complex::new(decode(fields[2]), decode(fields[3]));
                        let actual = LogValue::from_value(input).unwrap();
                        close(actual.log_magnitude().unwrap(), decode(fields[4]), 4);
                        let phase = actual.scaled_phase();
                        for (part, mi, ei) in [(phase.real(), 5, 6), (phase.imaginary(), 7, 8)] {
                            let expected = decode(fields[mi]);
                            if expected == 0.0 {
                                close(part.significand(), expected, 0);
                                continue;
                            }
                            let delta =
                                i128::from(part.exponent()) - fields[ei].parse::<i128>().unwrap();
                            assert!(delta.abs() <= 1, "{row}: phase exponent delta {delta}");
                            let aligned =
                                f64::from(part.significand()) * 2.0_f64.powi(delta as i32);
                            assert!(
                                (aligned / f64::from(expected) - 1.0).abs()
                                    <= 4.0 * f64::from(<$s>::EPSILON),
                                "{row}: phase {aligned:?} vs {expected:?}"
                            );
                        }
                    }
                    "project" => {
                        families[2] += 1;
                        let phase = Complex::new(decode(fields[2]), decode(fields[3]));
                        let encoded = LogValue::from_log(phase, decode(fields[4])).unwrap();
                        let expected = Complex::new(decode(fields[5]), decode(fields[6]));
                        if expected.re.is_infinite() || expected.im.is_infinite() {
                            assert!(
                                matches!(
                                    encoded.round().unwrap_err().view(),
                                    SciOxideErrorView::NumericalInstability {
                                        issue: NumericalIssue::Overflow,
                                        ..
                                    }
                                ),
                                "{row}"
                            );
                        } else {
                            let actual = encoded.round().unwrap();
                            close(actual.re, expected.re, 4);
                            close(actual.im, expected.im, 4);
                            if (phase.re != 0.0 && expected.re == 0.0)
                                || (phase.im != 0.0 && expected.im == 0.0)
                            {
                                assert!(
                                    matches!(
                                        encoded.try_value().unwrap_err().view(),
                                        SciOxideErrorView::NumericalInstability {
                                            issue: NumericalIssue::Underflow,
                                            ..
                                        }
                                    ),
                                    "{row}"
                                );
                            } else {
                                assert!(encoded.try_value().is_ok(), "{row}");
                            }
                        }
                    }
                    _ => unreachable!(),
                }
            }
            assert!(cases >= 90);
            assert!(
                families[0] >= 800 && families[1] >= 15 && families[2] >= 12 && families[3] >= 100,
                "incomplete corpus: {families:?}"
            );
        }
    };
}
oracle!(binary32_mpfr_mpc, f32, u32, "32");
oracle!(binary64_mpfr_mpc, f64, u64, "64");
