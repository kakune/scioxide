//! Closed-form regressions for complete spline contraction and rational range.
#![allow(clippy::unwrap_used)]
mod support;

use scioxide_core::{Complex, RealScalar, SciOxideErrorKind};
use scioxide_interp::{
    BSpline, BSplineBasis, Differentiable, ExtrapolationPolicy, Integrable, Interpolant, Nurbs,
    TwiceDifferentiable,
};
use support::{assert_close, from_f64};

fn line<S: RealScalar>(end: S) -> BSplineBasis<S> {
    BSplineBasis::new(vec![S::ZERO, S::ZERO, end, end], 1).unwrap()
}

fn close<S: RealScalar>(actual: S, expected: S) {
    assert_close(
        actual,
        expected,
        0.0,
        64.0 * S::EPSILON.try_to_f64_exact().unwrap(),
    );
    if expected != S::ZERO {
        assert!(actual != S::ZERO);
    }
}

fn tensor_range<S: RealScalar>(tiny: S, large: S, huge: S, derivative: S) {
    let reject = ExtrapolationPolicy::Reject;
    let basis = line(S::ONE);
    let model = BSpline::new(
        [basis.clone(), basis.clone()],
        vec![[S::ZERO; 2], [S::ZERO; 2], [S::ZERO; 2], [large, S::ONE]],
    )
    .unwrap();
    let mut workspace = model.workspace().unwrap();
    let mut out = [S::ZERO; 2];
    // The first component is representable after amplification; the second is
    // genuinely too small. Publication must not expose the first component.
    out.fill(S::ONE);
    assert!(
        model
            .evaluate_into([tiny; 2], reject, &mut out, &mut workspace)
            .is_err()
    );
    assert_eq!(out, [S::ONE; 2]);
    let scalar = BSpline::new(
        [basis.clone(), basis],
        vec![S::ZERO, S::ZERO, S::ZERO, large],
    )
    .unwrap();
    let mut scalar_ws = scalar.workspace().unwrap();
    let mut value = S::ZERO;
    {
        scalar
            .evaluate_into([tiny; 2], reject, &mut value, &mut scalar_ws)
            .unwrap();
    };
    close(value, tiny);

    let basis = line(S::ONE);
    let zero = Complex::new(S::ZERO, S::ZERO);
    let complex = BSpline::new(
        [basis.clone(), basis],
        vec![zero, zero, zero, Complex::new(large, large * S::HALF)],
    )
    .unwrap();
    let result = complex.evaluate([tiny; 2], reject).unwrap();
    close(result.re, tiny);
    close(result.im, tiny * S::HALF);

    let constant = BSplineBasis::new(vec![S::ZERO, tiny], 0).unwrap();
    let integral = BSpline::new([constant.clone(), constant], vec![large]).unwrap();
    close(
        integral.integral([S::ZERO; 2], [tiny; 2], reject).unwrap(),
        tiny,
    );
    let wide = line(large);
    let mut integral_ws = integral.workspace().unwrap();
    {
        integral
            .integral_into(
                [S::ZERO; 2],
                [tiny; 2],
                reject,
                &mut value,
                &mut integral_ws,
            )
            .unwrap();
    };
    close(value, tiny);
    let mixed = BSpline::new([wide.clone(), wide], vec![S::ZERO, S::ZERO, S::ZERO, huge]).unwrap();
    close(
        mixed
            .derivative([large * S::HALF; 2], [1, 1], None, reject)
            .unwrap(),
        derivative,
    );
    let mut mixed_ws = mixed.workspace().unwrap();
    {
        mixed
            .derivative_into(
                [large * S::HALF; 2],
                [1, 1],
                None,
                reject,
                &mut value,
                &mut mixed_ws,
            )
            .unwrap();
    };
    close(value, derivative);

    // The weight product overflows, then a small coefficient reduces it.
    let wide = BSplineBasis::new(vec![S::ZERO, large], 0).unwrap();
    let integral = BSpline::new([wide.clone(), wide], vec![tiny]).unwrap();
    close(
        integral.integral([S::ZERO; 2], [large; 2], reject).unwrap(),
        large,
    );
    let narrow = line(tiny);
    let mixed = BSpline::new(
        [narrow.clone(), narrow],
        vec![S::ZERO, S::ZERO, S::ZERO, tiny],
    )
    .unwrap();
    close(
        mixed
            .derivative([tiny * S::HALF; 2], [1, 1], None, reject)
            .unwrap(),
        large,
    );

    // Cancellation recovers a finite answer after an overflowing term/sum.
    let continued = BSpline::new([line(S::ONE)], vec![[S::MAX, tiny]; 2]).unwrap();
    let result = continued
        .evaluate([S::TWO], ExtrapolationPolicy::MethodDefined)
        .unwrap();
    close(result[0], S::MAX);
    close(result[1], tiny);
    let pieces = BSplineBasis::new(vec![S::ZERO, S::ONE, S::TWO, from_f64(3.0)], 0).unwrap();
    let cancelling = BSpline::new([pieces], vec![S::MAX, S::MAX, -S::MAX]).unwrap();
    close(
        cancelling
            .integral([S::ZERO], [from_f64(3.0)], reject)
            .unwrap(),
        S::MAX,
    );

    // Unit-weight NURBS consume the same tensor kernel before division.
    let basis = line(S::ONE);
    let rational = Nurbs::new(
        [basis.clone(), basis],
        vec![S::ZERO, S::ZERO, S::ZERO, large],
        vec![S::ONE; 4],
    )
    .unwrap();
    close(rational.evaluate([tiny; 2], reject).unwrap(), tiny);
    let mut workspace = rational.workspace().unwrap();
    {
        rational
            .evaluate_into([tiny; 2], reject, &mut value, &mut workspace)
            .unwrap();
    };
    close(value, tiny);
}

fn rational_line<S: RealScalar>(tiny: S, large: S) {
    let reject = ExtrapolationPolicy::Reject;
    for scale in [S::ONE, S::HALF, large] {
        let weights = vec![tiny * scale, scale];
        let controls = vec![[tiny, large], [S::TWO, from_f64(3.0)]];
        let vector = Nurbs::new([line(S::ONE)], controls.clone(), weights.clone()).unwrap();
        let complex = Nurbs::new(
            [line(S::ONE)],
            controls.iter().map(|c| Complex::new(c[0], c[1])).collect(),
            weights.clone(),
        )
        .unwrap();
        let real = Nurbs::new([line(S::ONE)], vec![tiny, S::TWO], weights).unwrap();
        let mut workspace = vector.workspace().unwrap();
        let mut out = [S::ZERO; 2];
        for t in [S::ZERO, S::MIN_POSITIVE, tiny, from_f64(0.25), S::ONE] {
            let a = (S::ONE - t) * tiny;
            let denominator = a + t;
            let expected = [
                tiny * (a / denominator) + S::TWO * (t / denominator),
                large * (a / denominator) + from_f64::<S>(3.0) * (t / denominator),
            ];
            {
                vector
                    .evaluate_into([t], reject, &mut out, &mut workspace)
                    .unwrap();
            };
            for c in 0..2 {
                close(out[c], expected[c]);
            }
            let z = complex.evaluate([t], reject).unwrap();
            assert_eq!([z.re, z.im], out);
            assert_eq!(real.evaluate([t], reject).unwrap(), out[0]);
            let expected_derivative = (tiny / denominator) * ((S::TWO - tiny) / denominator);
            close(real.gradient([t], reject).unwrap()[0], expected_derivative);
            if t >= from_f64(0.25) {
                let expected_second =
                    -S::TWO * ((S::ONE - tiny) / denominator) * expected_derivative;
                close(real.hessian([t], reject).unwrap()[0][0], expected_second);
                let second_component =
                    (tiny / denominator) * ((from_f64::<S>(3.0) - large) / denominator);
                let gradient = vector.gradient([t], reject).unwrap()[0];
                close(gradient[0], expected_derivative);
                close(gradient[1], second_component);
                let z = complex.gradient([t], reject).unwrap()[0];
                assert_eq!([z.re, z.im], gradient);
                let hessian = vector.hessian([t], reject).unwrap()[0][0];
                close(hessian[0], expected_second);
                close(
                    hessian[1],
                    -S::TWO * ((S::ONE - tiny) / denominator) * second_component,
                );
            }
        }
        // Values and first derivatives remain usable when second derivatives
        // cannot fit. A failed Hessian leaves output unchanged and scratch reusable.
        let mut real_ws = real.workspace().unwrap();
        let mut value = S::ONE;
        {
            real.gradient_into([S::ZERO], reject, &mut value, &mut real_ws)
                .unwrap();
            close(value, S::TWO / tiny);
            real.hessian_into([S::ONE], reject, &mut value, &mut real_ws)
                .unwrap();
            close(value, -from_f64::<S>(4.0) * tiny);
        };
        value = S::ONE;
        assert!(
            real.hessian_into([S::ZERO], reject, &mut value, &mut real_ws)
                .is_err()
        );
        assert_eq!(value, S::ONE);
        real.evaluate_into([S::ZERO], reject, &mut value, &mut real_ws)
            .unwrap();
        close(value, tiny);
        let mut pair = [S::ONE; 2];
        {
            assert!(
                vector
                    .gradient_into([S::ZERO], reject, &mut pair, &mut workspace)
                    .is_err()
            );
        };
        assert_eq!(pair, [S::ONE; 2]);
    }
    // Even a normalized weight below the scalar subnormal range is retained.
    let extreme = Nurbs::new(
        [line(S::ONE)],
        vec![tiny, S::TWO],
        vec![S::MIN_SUBNORMAL, S::MAX],
    )
    .unwrap();
    close(extreme.evaluate([S::ZERO], reject).unwrap(), tiny);
    let invalid =
        Nurbs::new([line(S::ONE)], vec![tiny, S::TWO], vec![S::ZERO, S::ONE]).unwrap_err();
    assert_eq!(invalid.kind(), SciOxideErrorKind::InvalidInput);
}

#[test]
fn contractions_retain_range_f64() {
    tensor_range(1e-200_f64, 1e200, 1e300, 1e-100);
}
#[test]
fn contractions_retain_range_f32() {
    tensor_range(1e-30_f32, 1e30, 1e35, 1e-25);
}
#[test]
fn homogeneous_controls_retain_range_f64() {
    rational_line(1e-200_f64, 1e200);
}
#[test]
fn homogeneous_controls_retain_range_f32() {
    rational_line(1e-30_f32, 1e30);
}

fn native_switchover<S: RealScalar>() {
    // Power-of-two coefficient scaling is exact, so results must scale exactly
    // (up to rounding) whether the native contraction or the range-safe one
    // admits a query; the sweep crosses both admission boundaries.
    let reject = ExtrapolationPolicy::Reject;
    let knots = [0., 0., 0., 0., 0.3, 0.7, 1., 1., 1., 1.]
        .map(from_f64::<S>)
        .to_vec();
    let basis = BSplineBasis::new(knots, 3).unwrap();
    let base: Vec<S> = (0..36)
        .map(|i| from_f64(((i * 7 % 11) as f64 - 5.) / 3.))
        .collect();
    let reference = BSpline::new([basis.clone(), basis.clone()], base.clone()).unwrap();
    let point = [from_f64::<S>(0.37), from_f64(0.81)];
    let value = reference.evaluate(point, reject).unwrap();
    let gradient = reference.gradient(point, reject).unwrap();
    let integral = reference
        .integral([from_f64(0.1); 2], [from_f64(0.9); 2], reject)
        .unwrap();
    let max = S::MAX.try_to_f64_exact().unwrap().log2() as i32;
    let min = (S::MIN_POSITIVE * S::EPSILON)
        .try_to_f64_exact()
        .unwrap()
        .log2() as i32;
    for exponent in (min + 8..max - 8).step_by(7) {
        let scale = S::TWO.powi(exponent);
        let scaled = BSpline::new(
            [basis.clone(), basis.clone()],
            base.iter().map(|&c| c * scale).collect(),
        )
        .unwrap();
        let check = |actual: Result<S, scioxide_core::SciOxideError>, expected: S| {
            let expected = expected * scale;
            match actual {
                Ok(actual) if expected.is_normal() => assert_close(
                    actual,
                    expected,
                    0.0,
                    64.0 * S::EPSILON.try_to_f64_exact().unwrap(),
                ),
                // Below the normal range a result may round or be rejected,
                // but never exceed rounding of the exact value.
                Ok(actual) => assert!((actual - expected).abs() <= S::MIN_POSITIVE),
                Err(error) => assert!(!expected.is_normal(), "{exponent}: {error:?}"),
            }
        };
        check(scaled.evaluate(point, reject), value);
        let scaled_gradient = scaled.gradient(point, reject);
        for axis in 0..2 {
            check(
                scaled_gradient
                    .as_ref()
                    .map(|g| g[axis])
                    .map_err(Clone::clone),
                gradient[axis],
            );
        }
        check(
            scaled.integral([from_f64(0.1); 2], [from_f64(0.9); 2], reject),
            integral,
        );
    }
}
#[test]
fn native_switchover_f32() {
    native_switchover::<f32>();
}
#[test]
fn native_switchover_f64() {
    native_switchover::<f64>();
}
