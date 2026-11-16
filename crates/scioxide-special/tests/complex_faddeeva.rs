//! Independent value/derivative envelopes and checked Faddeeva contracts.
#![allow(clippy::unwrap_used)]

use scioxide_core::{Budget, Complex, EstimateReliability, NumericScalar, RealScalar};
use scioxide_special::{
    ProbabilityOptions, ScalarRepresentation, SpecialFunctionTermination, faddeeva,
    faddeeva_with_options,
};

fn oracle<S: RealScalar>()
where
    Complex<S>: NumericScalar<Real = S>,
{
    for line in include_str!("data/complex_faddeeva.tsv")
        .lines()
        .filter(|l| !l.starts_with('#'))
    {
        let mut fields = line.split_whitespace();
        let width: u32 = fields.next().unwrap().parse().unwrap();
        if (width == 32) != (S::MANTISSA_DIGITS == 24) {
            continue;
        }
        let mut component = || {
            let bits = u64::from_str_radix(fields.next().unwrap(), 16).unwrap();
            let value = if width == 32 {
                f64::from(f32::from_bits(bits as u32))
            } else {
                f64::from_bits(bits)
            };
            if value.is_infinite() {
                if value.is_sign_negative() {
                    S::NEG_INFINITY
                } else {
                    S::INFINITY
                }
            } else {
                S::try_from_f64_lossy(value).unwrap()
            }
        };
        let z = Complex::new(component(), component());
        let value = Complex::new(component(), component());
        let derivative = Complex::new(component(), component());
        let report = faddeeva_with_options(z, ProbabilityOptions::default()).unwrap();
        assert_eq!(
            report.termination(),
            SpecialFunctionTermination::Converged,
            "{line}"
        );
        if value.re.is_finite() && value.im.is_finite() {
            let plain = faddeeva(z).unwrap();
            assert_eq!(plain, report.try_value().unwrap(), "{line}");
        } else {
            assert_eq!(
                report.scalar_representation(),
                ScalarRepresentation::Overflow,
                "{line}"
            );
            assert!(faddeeva(z).is_err(), "{line}");
            assert!(
                report.scaled().is_some() && report.logarithmic().is_some(),
                "{line}"
            );
        }
        for (actual, expected) in [(report.value(), value), (report.derivative(), derivative)] {
            if !expected.re.is_finite() || !expected.im.is_finite() {
                assert!(actual.is_none(), "{line}");
                continue;
            }
            assert!(actual.is_some(), "missing finite form: {line}");
            let actual = actual.unwrap();
            assert!(
                (*actual.value() - expected).norm() <= actual.absolute_error(),
                "{line}: actual={:?}, expected={expected:?}, error={:?}",
                actual.value(),
                actual.absolute_error()
            );
        }
    }
}

#[test]
fn oracle_f32() {
    oracle::<f32>();
}
#[test]
fn oracle_f64() {
    oracle::<f64>();
}

#[test]
fn completion_boundary_and_partial_reports_execute() {
    for z in [
        Complex::new(1e-5_f64, 0.5),
        Complex::new(3.0, 2.0),
        Complex::new(3.0, -2.0),
        Complex::new(27.0, 1e-12),
        Complex::new(9.0, 2.0),
        Complex::new(1.1, -18.0),
        Complex::new(0.3, -26.0),
    ] {
        let complete = faddeeva_with_options(z, ProbabilityOptions::default()).unwrap();
        let steps = complete.counts().iterations();
        assert!(steps > 1);
        for limit in [1, steps - 1, steps] {
            let options = ProbabilityOptions::new(Budget::try_from(limit).unwrap());
            let report = faddeeva_with_options(z, options).unwrap();
            assert_eq!(report.counts().iterations(), limit);
            if limit == steps {
                assert_eq!(report, complete);
            } else {
                assert_eq!(
                    report.termination(),
                    SpecialFunctionTermination::IterationLimit
                );
                assert_eq!(
                    report.value().unwrap().reliability(),
                    EstimateReliability::Heuristic
                );
                assert!(report.try_value().is_err());
            }
        }
        faddeeva(z).unwrap();
    }
}

#[test]
fn axis_signs_and_extended_reflection_remain_checked() {
    for zero in [0.0_f64, -0.0] {
        for y in [0.0, 2.0] {
            let value = faddeeva(Complex::new(zero, y)).unwrap();
            assert_eq!(value.im.to_bits(), zero.to_bits());
        }
    }
    for z in [Complex::new(1.0_f64, -20.0), Complex::new(1.0, -30.0)] {
        let report = faddeeva_with_options(z, ProbabilityOptions::default()).unwrap();
        assert_eq!(report.termination(), SpecialFunctionTermination::Converged);
        assert!(report.scaled().is_some() && report.logarithmic().is_some());
        if z.im == -20.0 {
            assert!(report.derivative().is_some());
            assert_eq!(report.scalar_representation(), ScalarRepresentation::Finite);
            assert_eq!(faddeeva(z).unwrap(), report.try_value().unwrap());
        } else {
            assert!(report.derivative().is_none());
            assert_eq!(
                report.scalar_representation(),
                ScalarRepresentation::Overflow
            );
            assert!(faddeeva(z).is_err() && report.try_value().is_err());
        }
    }
}

#[test]
fn binary32_scaled_reflection_allocates_nothing() {
    for y in [-8.0_f32, -9.0, -10.0] {
        let z = Complex::new(0.3, y);
        let report = faddeeva_with_options(z, ProbabilityOptions::default()).unwrap();
        assert!(report.scaled().is_some() && report.logarithmic().is_some());
        assert_eq!(report.derivative().is_some(), y > -10.0);
    }
}
