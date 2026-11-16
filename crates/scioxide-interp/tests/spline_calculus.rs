//! Independent polynomial and knot-side tests for spline calculus.
#![allow(clippy::unwrap_used)]
mod support;

use scioxide_core::{DomainReason, RealScalar, SciOxideError};
use scioxide_interp::{
    BSpline, BSplineBasis, Differentiable, ExtrapolationPolicy, Integrable, Interpolant, KnotSide,
};
use support::{assert_close, assert_error, from_f64};

fn spline<S: RealScalar>(knots: &[f64], control: &[f64], degree: usize) -> BSpline<S> {
    BSpline::new(
        [BSplineBasis::new(knots.iter().copied().map(from_f64).collect(), degree).unwrap()],
        control.iter().copied().map(from_f64).collect(),
    )
    .unwrap()
}

fn undefined<S>(result: Result<S, SciOxideError>) {
    assert!(matches!(
        assert_error(&result, scioxide_core::SciOxideErrorKind::DomainError).view(),
        scioxide_core::SciOxideErrorView::DomainError {
            reason: DomainReason::Undefined,
            ..
        }
    ));
}
fn knot_cases<S: RealScalar>() {
    let tolerance = 128. * <S as RealScalar>::EPSILON.into_f64();
    let reject = ExtrapolationPolicy::Reject;
    let mut step = spline::<S>(&[0., 1., 3.], &[2., -1.], 0);
    for (x, y) in [(0., 2.), (0.5, 2.), (1., -1.), (3., -1.)] {
        assert_close::<S, f64>(
            step.evaluate([from_f64(x)], reject).unwrap(),
            y,
            tolerance,
            tolerance,
        );
    }
    assert_close::<S, f64>(
        step.derivative([from_f64(0.5)], [1], None, reject).unwrap(),
        0.,
        tolerance,
        tolerance,
    );
    undefined(step.derivative([from_f64(1.)], [1], None, reject));
    for side in [KnotSide::Left, KnotSide::Right] {
        assert_close::<S, f64>(
            step.derivative([from_f64(1.)], [1], Some([side]), reject)
                .unwrap(),
            0.,
            tolerance,
            tolerance,
        );
    }
    assert_close::<S, f64>(
        step.derivative_spline()
            .unwrap()
            .evaluate([from_f64(1.)], reject)
            .unwrap(),
        0.,
        tolerance,
        tolerance,
    );
    step = step.insert_knot(from_f64(0.5)).unwrap();
    assert_close::<S, f64>(
        step.evaluate([from_f64(0.75)], reject).unwrap(),
        2.,
        tolerance,
        tolerance,
    );
    let before = step.bases()[0].knots().to_vec();
    assert!(step.insert_knot(from_f64(1.)).is_err());
    assert_eq!(step.bases()[0].knots(), before);
    let constant = spline::<S>(&[0., 1., 2.], &[3., 3.], 0);
    assert_close::<S, f64>(
        constant
            .derivative([from_f64(1.)], [1], None, reject)
            .unwrap(),
        0.,
        tolerance,
        tolerance,
    );
    let kink = spline::<S>(&[0., 0., 1., 2., 2.], &[0., 1., 0.], 1);
    undefined(kink.derivative([from_f64(1.)], [1], None, reject));
    assert_close::<S, f64>(
        kink.derivative([from_f64(1.)], [1], Some([KnotSide::Left]), reject)
            .unwrap(),
        1.,
        tolerance,
        tolerance,
    );
    assert_close::<S, f64>(
        kink.derivative([from_f64(1.)], [1], Some([KnotSide::Right]), reject)
            .unwrap(),
        -1.,
        tolerance,
        tolerance,
    );
    assert_close::<S, f64>(
        kink.derivative_spline()
            .unwrap()
            .evaluate([from_f64(0.5)], reject)
            .unwrap(),
        1.,
        tolerance,
        tolerance,
    );
    undefined(kink.derivative([from_f64(0.)], [1], Some([KnotSide::Left]), reject));
    assert_close::<S, f64>(
        kink.derivative([from_f64(0.)], [1], None, reject).unwrap(),
        1.,
        tolerance,
        tolerance,
    );
    assert_close::<S, f64>(
        kink.derivative(
            [from_f64(0.)],
            [1],
            Some([KnotSide::Left]),
            ExtrapolationPolicy::Clamp,
        )
        .unwrap(),
        0.,
        tolerance,
        tolerance,
    );
    assert_close::<S, f64>(
        kink.derivative(
            [from_f64(0.)],
            [1],
            Some([KnotSide::Left]),
            ExtrapolationPolicy::MethodDefined,
        )
        .unwrap(),
        1.,
        tolerance,
        tolerance,
    );
    let jump = spline::<S>(&[0., 0., 1., 1., 2., 2.], &[0., 1., 2., 3.], 1);
    undefined(jump.derivative([from_f64(1.)], [1], None, reject));
    let derivative = jump.derivative_spline().unwrap();
    for x in [0., 0.5, 1., 1.5, 2.] {
        assert_close::<S, f64>(
            derivative.evaluate([from_f64(x)], reject).unwrap(),
            1.,
            tolerance,
            tolerance,
        );
    }
    let smooth = spline::<S>(&[0., 0., 1., 1., 2., 2.], &[0., 1., 1., 2.], 1);
    assert_close::<S, f64>(
        smooth
            .derivative([from_f64(1.)], [1], None, reject)
            .unwrap(),
        1.,
        tolerance,
        tolerance,
    );
    // A workspace for a lower-degree model is too small for a cubic.
    let cubic = spline::<S>(&[0., 0., 0., 0., 1., 1., 1., 1.], &[0., 1., 2., 3.], 3);
    let mut empty = constant.workspace().unwrap();
    let mut out = S::ZERO;
    assert_error(
        &cubic.derivative_into(
            [from_f64(0.5)],
            [1],
            Some([KnotSide::Left]),
            reject,
            &mut out,
            &mut empty,
        ),
        scioxide_core::SciOxideErrorKind::DimensionMismatch,
    );
    assert!(smooth.derivative([S::NAN], [1], None, reject).is_err());
}
#[test]
fn knot_sides_f32() {
    knot_cases::<f32>();
}
#[test]
fn knot_sides_f64() {
    knot_cases::<f64>();
}

fn basis(knots: &[f64], i: usize, degree: usize, x: f64) -> f64 {
    if degree == 0 {
        return f64::from(knots[i] <= x && x < knots[i + 1]);
    }
    let left = knots[i + degree] - knots[i];
    let right = knots[i + degree + 1] - knots[i + 1];
    (if left == 0. {
        0.
    } else {
        (x - knots[i]) / left * basis(knots, i, degree - 1, x)
    }) + (if right == 0. {
        0.
    } else {
        (knots[i + degree + 1] - x) / right * basis(knots, i + 1, degree - 1, x)
    })
}
fn basis_cases<S: RealScalar>() {
    let tolerance = 128. * <S as RealScalar>::EPSILON.into_f64();
    for (knots, degree) in [
        (vec![0., 1., 3.], 0),
        (vec![0., 0., 0., 0.5, 0.5, 1., 2., 2., 2.], 2),
        (vec![-2., -1., 0., 1., 2., 3., 4.], 2),
    ] {
        let n = knots.len() - degree - 1;
        for j in 0..n {
            let mut controls = vec![0.; n];
            controls[j] = 1.;
            let mut s = spline::<S>(&knots, &controls, degree);
            let a = knots[degree];
            let b = knots[n];
            for k in 0..32 {
                let x = a + (b - a) * f64::from(k) / 32.;
                assert_close::<S, f64>(
                    s.evaluate([from_f64(x)], ExtrapolationPolicy::Reject)
                        .unwrap(),
                    basis(&knots, j, degree, x),
                    tolerance,
                    tolerance,
                );
            }
            let x = (a + b) * 0.5 + (b - a) / 64.;
            let expected = basis(&knots, j, degree, x);
            for _ in 0..=degree {
                s = s.insert_knot(from_f64(x)).unwrap();
            }
            assert_close::<S, f64>(
                s.evaluate([from_f64(x)], ExtrapolationPolicy::Reject)
                    .unwrap(),
                expected,
                tolerance,
                tolerance,
            );
        }
        let unity = spline::<S>(&knots, &vec![1.; n], degree);
        for x in [knots[degree], (knots[degree] + knots[n]) * 0.5, knots[n]] {
            assert_close::<S, f64>(
                unity
                    .evaluate([from_f64(x)], ExtrapolationPolicy::Reject)
                    .unwrap(),
                1.,
                tolerance,
                tolerance,
            );
        }
    }
    assert!(BSplineBasis::<S>::new(vec![S::ZERO, S::ZERO], 0).is_err());
    assert!(BSplineBasis::<S>::new(vec![S::ZERO, S::ZERO, S::ZERO, S::ONE, S::ONE], 1).is_err());
}
#[test]
fn basis_f32() {
    basis_cases::<f32>();
}
#[test]
fn basis_f64() {
    basis_cases::<f64>();
}

use scioxide_interp::{CubicSpline, SplineBoundary};
fn integral_cases<S: RealScalar>() {
    let tolerance = 128. * <S as RealScalar>::EPSILON.into_f64();
    let reject = ExtrapolationPolicy::Reject;
    // Bernstein coefficients for x^2 on [0,2].
    let quadratic = spline::<S>(&[0., 0., 0., 2., 2., 2.], &[0., 0., 4.], 2);
    let prepared = quadratic.prepare_antiderivative().unwrap();
    for (a, b) in [(0., 2.), (0.25, 1.75), (1., 1.), (1., 1.0001)] {
        let a = from_f64::<S>(a);
        let b = from_f64::<S>(b);
        let af = a.try_to_f64_exact().unwrap();
        let bf = b.try_to_f64_exact().unwrap();
        let expected = (bf - af) * (bf * bf + bf * af + af * af) / 3.;
        assert_close::<S, f64>(
            prepared.integral(a, b, reject).unwrap(),
            expected,
            tolerance,
            tolerance,
        );
        assert_close::<S, f64>(
            quadratic.integral([a], [b], reject).unwrap(),
            expected,
            tolerance,
            tolerance,
        );
    }
    assert_close::<S, f64>(
        prepared
            .integral(from_f64(-1.), from_f64(3.), ExtrapolationPolicy::Clamp)
            .unwrap(),
        20. / 3.,
        tolerance,
        tolerance,
    );
    assert_close::<S, f64>(
        prepared
            .integral(
                from_f64(-1.),
                from_f64(3.),
                ExtrapolationPolicy::MethodDefined,
            )
            .unwrap(),
        28. / 3.,
        tolerance,
        tolerance,
    );
    assert!(
        prepared
            .integral(from_f64(2.), from_f64(1.), reject)
            .is_err()
    );
    assert!(prepared.integral(S::NAN, from_f64(1.), reject).is_err());
    assert!(
        prepared
            .integral(from_f64(-1.), from_f64(-1.), reject)
            .is_err()
    );
    assert!(
        prepared
            .integral(
                from_f64(-1.),
                from_f64(1.),
                ExtrapolationPolicy::BoundedContinue {
                    max_distance: from_f64(0.5)
                }
            )
            .is_err()
    );
    assert_close::<S, f64>(
        prepared
            .integral(
                from_f64(-0.5),
                from_f64(1.),
                ExtrapolationPolicy::BoundedContinue {
                    max_distance: from_f64(0.5),
                },
            )
            .unwrap(),
        0.375,
        tolerance,
        tolerance,
    );
    let step = spline::<S>(&[0., 1., 3., 4.], &[2., -1., 3.], 0);
    let prepared = step.prepare_antiderivative().unwrap();
    assert_close::<S, f64>(
        prepared
            .integral(from_f64(0.5), from_f64(3.5), reject)
            .unwrap(),
        0.5,
        tolerance,
        tolerance,
    );
    assert_close::<S, f64>(
        prepared
            .integral(from_f64(0.), from_f64(4.), reject)
            .unwrap(),
        3.,
        tolerance,
        tolerance,
    );
    let knots = [0., 0., 0., 1., 1., 1., 2., 2., 2.];
    // x^2 then (x-1)^2 + 2, with a value jump at one.
    let mut jump = spline::<S>(&knots, &[0., 0., 1., 2., 2., 3.], 2);
    let snapshot = jump.prepare_antiderivative().unwrap();
    assert_close::<S, f64>(
        snapshot
            .integral(from_f64(0.), from_f64(2.), reject)
            .unwrap(),
        8. / 3.,
        tolerance,
        tolerance,
    );
    jump = jump.insert_knot(from_f64(0.5)).unwrap();
    assert_close::<S, f64>(
        jump.integral([from_f64(0.)], [from_f64(2.)], reject)
            .unwrap(),
        8. / 3.,
        tolerance,
        tolerance,
    );
    assert_close::<S, f64>(
        snapshot
            .integral(from_f64(0.), from_f64(2.), reject)
            .unwrap(),
        8. / 3.,
        tolerance,
        tolerance,
    );

    let nodes = [0., 0.25, 1., 2.];
    let x = nodes.iter().copied().map(from_f64).collect();
    let y = nodes
        .iter()
        .map(|x| from_f64(x * x * x - 2. * x + 1.))
        .collect();
    let cubic = CubicSpline::<S>::with_boundary(
        x,
        y,
        SplineBoundary::Clamped(from_f64(-2.), from_f64(10.)),
    )
    .unwrap();
    let prepared = cubic.prepare_antiderivative().unwrap();
    for (a, b) in [(0., 2.), (0.125, 1.5), (0.5, 0.75)] {
        let expected = (b * b * b * b - a * a * a * a) / 4. - (b * b - a * a) + b - a;
        assert_close::<S, f64>(
            prepared.integral(from_f64(a), from_f64(b), reject).unwrap(),
            expected,
            tolerance,
            tolerance,
        );
        assert_close::<S, f64>(
            cubic
                .integral([from_f64(a)], [from_f64(b)], reject)
                .unwrap(),
            expected,
            tolerance,
            tolerance,
        );
    }
    // Independent Cox-de Boor oracle integrated by composite Simpson. Each
    // interval is quadratic, so this rule is algebraically exact per piece.
    let knots = [-2., -1., 0., 0.5, 1., 2., 3., 4.];
    let controls = [1., -2., 3., -1., 2.];
    let s = spline::<S>(&knots, &controls, 2);
    let value = |x| {
        controls
            .iter()
            .enumerate()
            .map(|(i, &c)| c * basis(&knots, i, 2, x))
            .sum::<f64>()
    };
    let expected: f64 = [0., 0.5, 1., 2.]
        .windows(2)
        .map(|w| (w[1] - w[0]) * (value(w[0]) + 4. * value((w[0] + w[1]) / 2.) + value(w[1])) / 6.)
        .sum();
    assert_close::<S, f64>(
        s.integral([from_f64(0.)], [from_f64(2.)], reject).unwrap(),
        expected,
        tolerance,
        tolerance,
    );
}
#[test]
fn integrals_f32() {
    integral_cases::<f32>();
}
#[test]
fn integrals_f64() {
    integral_cases::<f64>();
}

#[test]
fn narrow_integral_avoids_primitive_subtraction() {
    let s = spline::<f64>(&[0., 0., 0., 1., 1., 1.], &[0., 0., 1.], 2);
    let p = s.prepare_antiderivative().unwrap();
    let a = 0.75_f64;
    let b = f64::from_bits(a.to_bits() + 1);
    let actual = p.integral(a, b, ExtrapolationPolicy::Reject).unwrap();
    let expected = (b - a) * (b * b + a * b + a * a) / 3.;
    assert!((actual - expected).abs() <= 4. * f64::EPSILON * expected.abs());
}

fn cubic_residuals<S: RealScalar>() {
    let comparison_tolerance = 128. * <S as RealScalar>::EPSILON.into_f64();
    let nodes = [0., 0.25, 1., 1.5, 2.];
    let values = [0., 1., -0.5, 0.25, 0.];
    let policies = [
        SplineBoundary::Natural,
        SplineBoundary::Clamped(from_f64(2.), from_f64(-1.)),
        SplineBoundary::NotAKnot,
        SplineBoundary::Periodic,
    ];
    for boundary in policies {
        let spline = CubicSpline::<S>::with_boundary(
            nodes.iter().copied().map(from_f64).collect(),
            values.iter().copied().map(from_f64).collect(),
            boundary,
        )
        .unwrap();
        let derivative = |x| {
            spline
                .gradient([from_f64(x)], ExtrapolationPolicy::Reject)
                .unwrap()[0]
                .try_to_f64_exact()
                .unwrap()
        };
        // Recover the piece's endpoint second derivatives independently from
        // three samples of its quadratic first derivative, in physical units.
        let sigmas: Vec<_> = nodes
            .windows(2)
            .map(|w| {
                let h = w[1] - w[0];
                let a = derivative(w[0]);
                let m = derivative((w[0] + w[1]) / 2.);
                let b = derivative(w[1]);
                ((-3. * a + 4. * m - b) / h, (a - 4. * m + 3. * b) / h)
            })
            .collect();
        let tolerance = 8192. * S::EPSILON.try_to_f64_exact().unwrap();
        for i in 1..nodes.len() - 1 {
            let h0 = nodes[i] - nodes[i - 1];
            let h1 = nodes[i + 1] - nodes[i];
            assert!((sigmas[i - 1].1 - sigmas[i].0).abs() < tolerance);
            let lhs = h0 * sigmas[i - 1].0 + 2. * (h0 + h1) * sigmas[i].0 + h1 * sigmas[i].1;
            let rhs = 6. * ((values[i + 1] - values[i]) / h1 - (values[i] - values[i - 1]) / h0);
            assert!((lhs - rhs).abs() <= tolerance * rhs.abs().max(1.));
        }
        match boundary {
            SplineBoundary::Natural => {
                assert!(sigmas[0].0.abs() < tolerance);
                assert!(sigmas[3].1.abs() < tolerance);
            }
            SplineBoundary::Clamped(..) => {
                assert_close::<S, f64>(
                    from_f64::<S>(derivative(0.)),
                    2.,
                    comparison_tolerance,
                    comparison_tolerance,
                );
                assert_close::<S, f64>(
                    from_f64::<S>(derivative(2.)),
                    -1.,
                    comparison_tolerance,
                    comparison_tolerance,
                );
            }
            SplineBoundary::NotAKnot => {
                for (i, j) in [(0, 1), (2, 3)] {
                    let a = (sigmas[i].1 - sigmas[i].0) / (nodes[i + 1] - nodes[i]);
                    let b = (sigmas[j].1 - sigmas[j].0) / (nodes[j + 1] - nodes[j]);
                    assert!((a - b).abs() < tolerance * a.abs().max(1.));
                }
            }
            SplineBoundary::Periodic => {
                assert!((derivative(0.) - derivative(2.)).abs() < tolerance);
                assert!((sigmas[0].0 - sigmas[3].1).abs() < tolerance);
            }
        }
    }
}
#[test]
fn shared_factor_residuals_f32() {
    cubic_residuals::<f32>();
}
#[test]
fn shared_factor_residuals_f64() {
    cubic_residuals::<f64>();
}

#[test]
fn high_degree_monomial_integrals() {
    let p = 8;
    let knots: Vec<_> = core::iter::repeat_n(0., p + 1)
        .chain(core::iter::repeat_n(1., p + 1))
        .collect();
    for power in 0..=p {
        let controls: Vec<_> = (0..=p)
            .map(|i| {
                if i < power {
                    return 0.;
                }
                (0..power)
                    .map(|j| (i - j) as f64 / (p - j) as f64)
                    .product()
            })
            .collect();
        let spline = spline::<f64>(&knots, &controls, p);
        let actual = spline
            .integral([0.], [1.], ExtrapolationPolicy::Reject)
            .unwrap();
        assert!((actual - 1. / (power + 1) as f64).abs() < 1e-12);
    }
}

#[test]
fn endpoints_and_integrals_preserve_extreme_values() {
    let linear = spline::<f64>(&[0., 0., 1., 1.], &[1e20, 1.], 1);
    assert_eq!(
        linear.evaluate([1.], ExtrapolationPolicy::Reject).unwrap(),
        1.
    );
    let constant = spline::<f64>(
        &[-f64::MAX, -f64::MAX, f64::MAX, f64::MAX],
        &[0.25, 0.25],
        1,
    );
    assert_eq!(
        constant
            .integral([-f64::MAX], [f64::MAX], ExtrapolationPolicy::Reject)
            .unwrap(),
        f64::MAX / 2.
    );
}

#[test]
fn unclamped_endpoint_insertion_preserves_the_whole_domain() {
    let tolerance = 128. * <f64 as RealScalar>::EPSILON.into_f64();
    let source = spline::<f64>(&[-2., -1., 0., 1., 2., 3., 4.], &[1., -2., 3., -1.], 2);
    for knot in [0., 2.] {
        let mut refined = source.clone();
        for _ in 0..2 {
            refined = refined.insert_knot(knot).unwrap();
            for i in 0..=16 {
                let x = f64::from(i) / 8.;
                assert_close::<f64, f64>(
                    refined.evaluate([x], ExtrapolationPolicy::Reject).unwrap(),
                    source.evaluate([x], ExtrapolationPolicy::Reject).unwrap(),
                    tolerance,
                    tolerance,
                );
            }
        }
    }
}
