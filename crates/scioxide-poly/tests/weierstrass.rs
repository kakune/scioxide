//! The shared Weierstrass correction kernel.
#![allow(clippy::unwrap_used)]

use scioxide_core::{Complex, RealScalar, ScaledValue};
use scioxide_poly::backend::polynomial_iteration::{WeierstrassCorrection, weierstrass_correction};

type Scaled<S> = ScaledValue<Complex<S>>;

fn scaled<S: RealScalar>(outcome: WeierstrassCorrection<S>) -> Option<(Scaled<S>, Scaled<S>)> {
    match outcome {
        WeierstrassCorrection::Defined {
            correction,
            product,
        } => Some((correction, product)),
        WeierstrassCorrection::Coincident => None,
    }
}

fn weierstrass_rule<S: RealScalar>() {
    let c = |re: f64, im: f64| {
        Complex::new(
            S::try_from_f64_lossy(re).unwrap(),
            S::try_from_f64_lossy(im).unwrap(),
        )
    };
    let correction = |point: Complex<S>, others: &[Complex<S>], value: Complex<S>| {
        weierstrass_correction(
            point,
            others.iter().copied(),
            ScaledValue::from_value(value).unwrap(),
        )
        .unwrap()
    };
    let defined = |outcome: WeierstrassCorrection<S>| {
        let (correction, product) = scaled(outcome).unwrap();
        (correction.round().unwrap(), product.round().unwrap())
    };
    // The only undefined case is an exactly equal candidate, including +0/-0
    // components; the result is the same wherever the duplicate sits.
    let point = c(0.5, -0.25);
    for others in [
        [point, c(2.0, 0.0)],
        [c(2.0, 0.0), point],
        [c(2.0, 0.0), c(0.5, -0.25)],
    ] {
        assert_eq!(
            correction(point, &others, c(1.0, 0.0)),
            WeierstrassCorrection::Coincident
        );
    }
    assert_eq!(
        correction(c(0.0, 0.0), &[c(-0.0, -0.0)], c(1.0, 0.0)),
        WeierstrassCorrection::Coincident
    );
    let big = Complex::new(S::MAX * S::HALF, S::ZERO);
    assert_eq!(
        correction(big, &[-big, big], c(1.0, 0.0)),
        WeierstrassCorrection::Coincident
    );
    // Candidates one ulp apart are distinct: their exact difference is used.
    let one = c(1.0, 0.0);
    let next = Complex::new(S::ONE + S::EPSILON, S::ZERO);
    let (w, product) = defined(correction(one, &[next], Complex::new(S::EPSILON, S::ZERO)));
    assert_eq!(product, Complex::new(-S::EPSILON, S::ZERO));
    assert!((w - c(-1.0, 0.0)).norm() <= S::EPSILON);
    // Ordinary corrections: p(z) = z^3 - 1 with candidates 2, i, -i gives
    // w = 7 / ((2 - i)(2 + i)) = 7 / 5, in either product order.
    let others = [c(0.0, 1.0), c(0.0, -1.0)];
    let reversed = [others[1], others[0]];
    for others in [others, reversed] {
        let (w, product) = defined(correction(c(2.0, 0.0), &others, c(7.0, 0.0)));
        let tolerance = S::try_from_f64_lossy(4.0).unwrap() * S::EPSILON;
        assert!((w - c(1.4, 0.0)).norm() <= tolerance);
        assert!((product - c(5.0, 0.0)).norm() <= tolerance);
    }
    // Products beyond the native range retry in extended range.
    let huge = S::MAX.sqrt() * S::TWO;
    let outcome = correction(
        Complex::new(huge, S::ZERO),
        &[Complex::new(-huge, S::ZERO), Complex::new(S::ZERO, huge)],
        c(1.0, 0.0),
    );
    let (w, product) = scaled(outcome).unwrap();
    assert!(product.round().is_err());
    // w = 1 / (2h (h - ih)) = (1 + i) / (4 h^2).
    let h = ScaledValue::from_value(Complex::new(huge, S::ZERO)).unwrap();
    let expected = ScaledValue::from_value(c(0.25, 0.25))
        .unwrap()
        .div(h.mul(h).unwrap())
        .unwrap();
    let error = w.sub(expected).unwrap().magnitude().unwrap();
    let bound = expected
        .magnitude()
        .unwrap()
        .mul(
            scioxide_core::Scaled::from_value(S::try_from_f64_lossy(8.0).unwrap() * S::EPSILON)
                .unwrap(),
        )
        .unwrap();
    assert!(error.abs_le(bound));
    // Nonfinite candidates are rejected rather than treated as coincident.
    assert!(
        weierstrass_correction(
            one,
            [Complex::new(S::NAN, S::ZERO)].into_iter(),
            ScaledValue::from_value(one).unwrap(),
        )
        .is_err()
    );
}

#[test]
fn weierstrass_correction_has_one_exact_coincidence_rule_in_both_widths() {
    weierstrass_rule::<f32>();
    weierstrass_rule::<f64>();
}

fn nonfinite_points<S: RealScalar>() {
    let zero = Complex::new(S::ZERO, S::ZERO);
    let one = Complex::new(S::ONE, S::ZERO);
    let value = ScaledValue::from_value(one).unwrap();
    for component in [S::NAN, S::INFINITY, -S::INFINITY] {
        for point in [
            Complex::new(component, S::ZERO),
            Complex::new(S::ZERO, component),
        ] {
            for others in [&[][..], &[zero][..], &[point][..]] {
                let error =
                    weierstrass_correction(point, others.iter().copied(), value).unwrap_err();
                assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::NonFinite);
            }
        }
    }
}

#[test]
fn weierstrass_rejects_nonfinite_points_even_without_other_candidates() {
    nonfinite_points::<f32>();
    nonfinite_points::<f64>();
}

fn subnormal_quotients<S: RealScalar>(small: S, large: S) {
    let zero = S::ZERO;
    let point = Complex::new(small, zero);
    let other = Complex::new(-large, zero);
    let factor = S::try_from_f64_lossy(1.234567890123).unwrap();
    // Real, imaginary and mixed numerators exercise each quotient component.
    for numerator in [
        Complex::new(small * factor, zero),
        Complex::new(zero, small * factor),
        Complex::new(small * factor, small),
        Complex::new(large, small * factor),
    ] {
        let value = ScaledValue::from_value(numerator).unwrap();
        let (correction, product) =
            scaled(weierstrass_correction(point, [other].into_iter(), value).unwrap()).unwrap();
        let expected_product = ScaledValue::from_value(point)
            .unwrap()
            .sub(ScaledValue::from_value(other).unwrap())
            .unwrap();
        let expected = value.div(expected_product).unwrap();
        for (actual, expected) in [
            (correction.real(), expected.real()),
            (correction.imaginary(), expected.imaginary()),
        ] {
            let error = actual.sub(expected).unwrap().abs();
            let bound = expected
                .abs()
                .mul(
                    scioxide_core::Scaled::from_value(
                        S::try_from_f64_exact(4.0).unwrap() * S::EPSILON,
                    )
                    .unwrap(),
                )
                .unwrap();
            assert!(error.abs_le(bound), "{actual:?} != {expected:?}");
        }
        assert_eq!(product, expected_product);
    }
}

#[test]
fn weierstrass_retains_subnormal_quotients_in_extended_range() {
    subnormal_quotients::<f32>(1e-30, 1e10);
    subnormal_quotients::<f64>(1e-300, 1e10);
    subnormal_quotients::<f32>(1e-30, 1e30);
    subnormal_quotients::<f64>(1e-300, 1e300);
}
