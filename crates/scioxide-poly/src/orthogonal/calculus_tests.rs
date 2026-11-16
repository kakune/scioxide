//! Independent dense differentiated-recurrence oracle for compact calculus.
use super::*;
use crate::PolynomialBasis;
use crate::{ApproxPolynomial, PolynomialApproximationPolicy};
use alloc::vec::Vec;
use scioxide_core::{AlgebraicScalar, Complex, NumericScalar, RealScalar};

fn dense<C: NumericScalar>(plan: &OrthogonalBasisPlan<C::Real>, coefficients: &[C]) -> Vec<C> {
    let rows = plan.degree().max(1);
    let mut previous = vec![C::ZERO; rows];
    let mut current = previous.clone();
    let mut next = previous.clone();
    let mut result = previous.clone();
    for n in 0..plan.degree() {
        next.fill(C::ZERO);
        for (k, &value) in current.iter().enumerate() {
            let (a, b, c) = plan.root_recurrence(k);
            let factor = value / C::from_real(a);
            next[k] = next[k] - factor * C::from_real(b);
            if k + 1 < rows {
                next[k + 1] = next[k + 1] + factor;
            }
            if k > 0 {
                next[k - 1] = next[k - 1] + factor * C::from_real(c);
            }
        }
        let (a, b, c) = plan.root_recurrence(n);
        for (k, out) in next.iter_mut().enumerate() {
            *out = *out * C::from_real(a) + current[k] * C::from_real(b)
                - previous[k] * C::from_real(c);
            if k == n {
                *out = *out + C::from_real(a);
            }
            result[k] = result[k] + coefficients[n + 1] * *out;
        }
        core::mem::swap(&mut previous, &mut current);
        core::mem::swap(&mut current, &mut next);
    }
    result
}
fn near<C: NumericScalar>(actual: &[C], expected: &[C], context: &str) {
    assert_eq!(actual.len(), expected.len());
    let scale = expected
        .iter()
        .map(|c| c.abs())
        .fold(C::Real::ONE, |a, b| a.max(b));
    let tolerance = C::Real::try_from_f64_exact(2048.0).unwrap() * C::Real::EPSILON * scale;
    for (&a, &b) in actual.iter().zip(expected) {
        assert!(
            (a - b).abs() <= tolerance,
            "{context}: {a:?} != {b:?}, tolerance {tolerance:?}"
        );
    }
}
fn exercise<C: NumericScalar>(imaginary: f64) {
    let r = |value| C::Real::try_from_f64_lossy(value).unwrap();
    let families = [
        OrthogonalFamily::ChebyshevT,
        OrthogonalFamily::ChebyshevU,
        OrthogonalFamily::Legendre,
        OrthogonalFamily::Jacobi {
            alpha: r(0.25),
            beta: r(0.75),
        },
        OrthogonalFamily::Jacobi {
            alpha: r(-0.75),
            beta: r(-0.25),
        },
        OrthogonalFamily::Jacobi {
            alpha: r(0.25),
            beta: r(-0.25),
        },
        OrthogonalFamily::Gegenbauer { lambda: r(1.5) },
        OrthogonalFamily::Gegenbauer { lambda: r(-0.25) },
        OrthogonalFamily::Laguerre,
        OrthogonalFamily::GeneralizedLaguerre { alpha: r(-0.75) },
        OrthogonalFamily::HermiteH,
        OrthogonalFamily::HermiteHe,
    ];
    for family in families {
        for normalization in [
            BasisNormalization::Classical,
            BasisNormalization::Monic,
            BasisNormalization::Orthonormal,
        ] {
            for (shift, scale) in [(0.0, 1.0), (0.25, 1.5)] {
                let spec =
                    OrthogonalBasisSpec::new(family, normalization, r(shift), r(scale)).unwrap();
                for degree in [0, 1, 2, 8] {
                    let coefficients: Vec<_> = (0..=degree)
                        .map(|k| {
                            let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
                            // Dyadic decay keeps the dense oracle well conditioned even
                            // for monic Laguerre's factorial coefficient growth.
                            let weight = (0..k).fold(C::Real::ONE, |weight, _| weight * r(0.25));
                            C::from_components(r(sign) * weight, r(imaginary) * weight)
                        })
                        .collect();
                    let polynomial = ApproxPolynomial::orthogonal_with_plan(
                        coefficients.clone(),
                        spec.prepare(coefficients.len() - 1, Default::default())
                            .unwrap(),
                        PolynomialApproximationPolicy::default(),
                    )
                    .unwrap();
                    let context = format!(
                        "{family:?}, {normalization:?}, shift {shift}, degree {degree}, width {}",
                        C::Real::EPSILON.into_f64()
                    );
                    let actual = polynomial.derivative().unwrap();
                    near(
                        actual.basis().coefficients(),
                        &dense(polynomial.basis().plan(), &coefficients),
                        &context,
                    );
                    let primitive = polynomial
                        .antiderivative(C::from_real(r(0.125)), C::from_real(r(0.5)))
                        .unwrap();
                    let output_plan = primitive.basis().plan();
                    let mut reference = vec![C::ZERO; degree + 2];
                    let columns: Vec<_> = (0..degree + 2)
                        .map(|column| {
                            let mut unit = vec![C::ZERO; degree + 2];
                            unit[column] = C::ONE;
                            dense(output_plan, &unit)
                        })
                        .collect();
                    for row in (0..=degree).rev() {
                        let tail = (row + 2..degree + 2).fold(C::ZERO, |sum, column| {
                            sum + columns[column][row] * reference[column]
                        });
                        reference[row + 1] = (coefficients[row] - tail) / columns[row + 1][row];
                    }
                    near(
                        &primitive.basis().coefficients()[1..],
                        &reference[1..],
                        &context,
                    );
                    near(
                        &dense(output_plan, primitive.basis().coefficients()),
                        &coefficients,
                        &context,
                    );
                    let evaluation = primitive
                        .evaluate_with_derivative(C::from_real(r(0.125)))
                        .unwrap();
                    let tolerance = r(128.0)
                        * C::Real::EPSILON
                        * (C::Real::ONE + evaluation.condition().condition_estimate());
                    assert!(
                        (evaluation.value() - C::from_real(r(0.5))).abs() <= tolerance,
                        "{family:?}, {normalization:?}, degree {degree}"
                    );
                }
            }
        }
    }
}
#[test]
fn compact_calculus_matches_dense_all_widths() {
    exercise::<f32>(0.0);
    exercise::<f64>(0.0);
    exercise::<Complex<f32>>(0.125);
    exercise::<Complex<f64>>(0.125);
}
