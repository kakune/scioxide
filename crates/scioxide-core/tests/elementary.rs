//! Independent MPFR fixtures exercise the actual public scalar dispatch.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
use scioxide_core::{Complex, RealScalar};

macro_rules! oracle_test {
    ($name:ident, $ty:ty, $bits:ty, $width:literal) => {
        #[test]
        fn $name() {
            let mut failures = Vec::new();
            let mut maxima = std::collections::BTreeMap::<&str, u64>::new();
            for line in include_str!("data/elementary.txt")
                .lines()
                .filter(|l| !l.starts_with('#'))
            {
                let row: Vec<_> = line.split_whitespace().collect();
                if row[0] != $width {
                    continue;
                }
                let decode = |s| <$ty>::from_bits(<$bits>::from_str_radix(s, 16).unwrap());
                let x = decode(row[2]);
                let y = decode(row[3]);
                let expected = decode(row[4]);
                let actual = match row[1] {
                    "sin" => RealScalar::sin(x),
                    "cos" => RealScalar::cos(x),
                    "tan" => RealScalar::tan(x),
                    "asin" => RealScalar::asin(x),
                    "acos" => RealScalar::acos(x),
                    "atan" => RealScalar::atan(x),
                    "sinh" => RealScalar::sinh(x),
                    "cosh" => RealScalar::cosh(x),
                    "tanh" => RealScalar::tanh(x),
                    "asinh" => RealScalar::asinh(x),
                    "acosh" => RealScalar::acosh(x),
                    "atanh" => RealScalar::atanh(x),
                    "exp" => RealScalar::exp(x),
                    "exp2" => RealScalar::exp2(x),
                    "exp10" => RealScalar::exp10(x),
                    "exp_m1" => RealScalar::exp_m1(x),
                    "ln" => RealScalar::ln(x),
                    "ln_1p" => RealScalar::ln_1p(x),
                    "log2" => RealScalar::log2(x),
                    "log10" => RealScalar::log10(x),
                    "cbrt" => RealScalar::cbrt(x),
                    "hypot" => RealScalar::hypot(x, y),
                    "atan2" => RealScalar::atan2(x, y),
                    "powf" => RealScalar::powf(x, y),
                    "log" => RealScalar::log(x, y),
                    other => panic!("{other}"),
                };
                let distance = actual.to_bits().abs_diff(expected.to_bits()) as u64;
                let valid = if expected.is_nan() {
                    actual.is_nan()
                } else if !expected.is_finite() || expected == 0.0 {
                    actual.to_bits() == expected.to_bits()
                } else {
                    actual.classify() == expected.classify()
                        && actual.is_sign_negative() == expected.is_sign_negative()
                        && distance <= if row[1] == "log" { 16 } else { 4 }
                };
                if expected.is_finite() && actual.is_finite() {
                    maxima
                        .entry(row[1])
                        .and_modify(|v| *v = (*v).max(distance))
                        .or_insert(distance);
                }
                if !valid {
                    failures.push(format!(
                        "{line}: {x:?}, {y:?} -> {actual:?}, want {expected:?} ({distance} ULP)"
                    ));
                }
                if row[1] == "powf"
                    && y.is_finite()
                    && y >= i32::MIN as $ty
                    && (y as f64) <= f64::from(i32::MAX)
                    && y.fract() == 0.0
                {
                    let powered = RealScalar::powi(x, y as i32);
                    assert!(
                        powered.is_nan() && expected.is_nan()
                            || powered.classify() == expected.classify()
                                && powered.is_sign_negative() == expected.is_sign_negative()
                                && if expected == 0.0 || expected.is_infinite() {
                                    powered.to_bits() == expected.to_bits()
                                } else {
                                    powered.to_bits().abs_diff(expected.to_bits()) <= 4
                                },
                        "powi: {line}: {powered:?}, want {expected:?}"
                    );
                }
                if row[1] == "sin" || row[1] == "cos" {
                    let pair = RealScalar::sin_cos(x);
                    let paired = if row[1] == "sin" { pair.0 } else { pair.1 };
                    assert!(
                        paired.is_nan() && actual.is_nan() || paired.to_bits() == actual.to_bits(),
                        "sin_cos: {line}"
                    );
                }
            }
            eprintln!("{} maxima: {maxima:?}", $width);
            assert!(
                failures.is_empty(),
                "{} failures:\n{}",
                failures.len(),
                failures
                    .iter()
                    .take(25)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join("\n")
            );
        }
    };
}
oracle_test!(binary32_oracle, f32, u32, "32");
oracle_test!(binary64_oracle, f64, u64, "64");

#[test]
fn complex_extreme_regressions() {
    for z in [
        Complex::new(f64::MAX, f64::MAX),
        Complex::new(1e-300, 1e-300),
    ] {
        assert!(z.sqrt().is_finite());
        assert!(z.ln().is_finite());
        assert!(z.asinh().is_finite());
        assert!(z.asin().is_finite());
        assert!(z.acos().is_finite());
        assert!(z.acosh().is_finite());
    }
    assert!((Complex::new(1.0_f64, 1000.0).tan().im - 1.0).abs() < 1e-15);
    assert!((Complex::new(1000.0_f64, 1.0).tanh().re - 1.0).abs() < 1e-15);
    assert_eq!(Complex::new(0.0, 0.0).powf(0.0), Complex::new(1.0, 0.0));
    assert!(RealScalar::powi(1e20_f64, -2) > 0.0);
}

macro_rules! complex_oracle {
    ($name:ident,$ty:ty,$bits:ty,$width:literal) => {
        #[test]
        fn $name() {
            let mut failures=Vec::new();
            for line in include_str!("data/complex_elementary.txt").lines().filter(|l| !l.starts_with('#')) {
                let r:Vec<_>=line.split_whitespace().collect();
                if r[0]!=$width {continue;}
                let decode=|s| <$ty>::from_bits(<$bits>::from_str_radix(s,16).unwrap());
                let z=Complex::new(decode(r[2]),decode(r[3]));
                let w=Complex::new(decode(r[4]),decode(r[5]));
                let result=match r[1] {
                    "sqrt"=>z.sqrt(),"exp"=>z.exp(),"log"=>z.ln(),
                    "sin"=>z.sin(),"cos"=>z.cos(),"tan"=>z.tan(),
                    "sinh"=>z.sinh(),"cosh"=>z.cosh(),"tanh"=>z.tanh(),
                    "asin"=>z.asin(),"acos"=>z.acos(),"atan"=>z.atan(),
                    "asinh"=>z.asinh(),"acosh"=>z.acosh(),"atanh"=>z.atanh(),
                    "powf"=>z.powf(w.re),"powc"=>z.powc(w),"powi"=>z.powi(3),"powu"=>z.powu(3),op=>panic!("{op}")
                };
                for (index,actual) in [result.re,result.im].into_iter().enumerate() {
                    let expected=decode(r[6+index]);
                    let bound=f64::from_bits(u64::from_str_radix(r[8+index],16).unwrap());
                    let valid=if expected.is_nan(){actual.is_nan()}
                    else if expected.is_infinite(){actual==expected}
                    else if expected==0.0 {(actual.to_bits()==expected.to_bits() || r[1].starts_with("pow") && actual==0.0) || actual.is_finite() && (actual as f64).abs()<=bound && actual!=0.0}
                    else {actual.is_finite() && ((actual as f64)-(expected as f64)).abs()<=bound};
                    if !valid {failures.push(format!("{} {:?}: component {index}: {actual:?}, want {expected:?}, bound {bound:e}",r[1],z));}
                }
            }
            assert!(failures.is_empty(),"{} failures:\n{}",failures.len(),failures.iter().take(45).cloned().collect::<Vec<_>>().join("\n"));
        }
    }
}
complex_oracle!(complex32_oracle, f32, u32, "32");
complex_oracle!(complex64_oracle, f64, u64, "64");

#[test]
fn complex_nonfinite_domains_and_branch_sides() {
    for zero in [0.0_f64, -0.0] {
        let negative_axis = Complex::new(-2.0, zero);
        assert_eq!(
            negative_axis.ln().im.is_sign_negative(),
            zero.is_sign_negative()
        );
        assert_eq!(
            negative_axis.sqrt().im.is_sign_negative(),
            zero.is_sign_negative()
        );
        assert_eq!(
            negative_axis.asin().im.is_sign_negative(),
            zero.is_sign_negative()
        );
        assert_eq!(
            negative_axis.acos().im.is_sign_negative(),
            !zero.is_sign_negative()
        );
        let root = Complex::new(f64::INFINITY, zero).sqrt();
        assert_eq!(root.re, f64::INFINITY);
        assert_eq!(root.im.to_bits(), zero.to_bits());
        let exponential = Complex::new(f64::INFINITY, zero).exp();
        assert_eq!(exponential.re, f64::INFINITY);
        assert_eq!(exponential.im.to_bits(), zero.to_bits());
    }
    for infinity in [f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(Complex::new(1.0, infinity).sqrt().im, infinity);
        assert_eq!(Complex::new(infinity, 1.0).ln().re, f64::INFINITY);
        assert_eq!(
            Complex::new(1.0, infinity).tan().im,
            1.0_f64.copysign(infinity)
        );
        assert_eq!(
            Complex::new(infinity, 1.0).tanh().re,
            1.0_f64.copysign(infinity)
        );
    }
    let nan = Complex::new(f64::NAN, f64::NAN);
    for result in [
        nan.sqrt(),
        nan.exp(),
        nan.ln(),
        nan.sin(),
        nan.cos(),
        nan.tan(),
        nan.sinh(),
        nan.cosh(),
        nan.tanh(),
        nan.asin(),
        nan.acos(),
        nan.atan(),
        nan.asinh(),
        nan.acosh(),
        nan.atanh(),
    ] {
        assert!(result.re.is_nan() && result.im.is_nan());
    }
}

#[test]
fn independent_identities_and_integer_power_extremes() {
    for x in [-1e20_f64, -3.0, -0.01, 0.01, 3.0, 1e20] {
        let (s, c) = RealScalar::sin_cos(x);
        assert!((s * s + c * c - 1.0).abs() <= 4.0 * f64::EPSILON);
        assert!((RealScalar::asinh(x).sinh() - x).abs() <= 32.0 * f64::EPSILON * x.abs());
    }
    assert_eq!(RealScalar::powi(2.0_f64, -1074).to_bits(), 1);
    assert_eq!(RealScalar::powi(2.0_f32, -149).to_bits(), 1);
    assert_eq!(RealScalar::powi(-1.0_f64, i32::MIN), 1.0);
    let z = Complex::new(f64::MAX, f64::from_bits(1));
    let cube = z.powi(3);
    assert!(cube.re.is_infinite() && cube.im.is_finite() && cube.im > 0.0);
}

#[test]
fn complex_powers_reduce_phases_beyond_the_scalar_range() {
    for z in [
        Complex::new(-1.0_f64, 0.0),
        Complex::new(0.0, 1.0),
        Complex::new(20.0, 0.0),
    ] {
        let result = z.powc(Complex::new(0.0, f64::MAX));
        assert!(result.is_finite(), "{z:?}: {result:?}");
    }
    assert!(Complex::new(0.0_f64, 1.0).powf(f64::MAX).is_finite());
    assert!(Complex::new(0.0_f32, 1.0).powf(f32::MAX).is_finite());
    assert!(Complex::new(1e200_f64, 1e-200).powi(-2).is_finite());
}
