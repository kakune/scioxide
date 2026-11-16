//! Public orthogonal representation and coefficient calculus contracts.
#![allow(clippy::unwrap_used)]

#[path = "support/tolerance.rs"]
mod tolerance;
use scioxide_core::AlgebraicScalar;
#[cfg(feature = "std")]
use scioxide_core::ExecutionPolicy;
use scioxide_core::{Complex, ExecutionContext, LinearOperator, NumericScalar, RealScalar};
use scioxide_poly::PolynomialBasis;
use scioxide_poly::{
    ApproxPolynomial, BasisNormalization as N, OrthogonalBasisSpec, OrthogonalFamily as F,
    PolynomialApproximationPolicy,
};
use tolerance::assert_close;

fn families<R: RealScalar>() -> Vec<F<R>> {
    let half = R::try_from_f64_exact(0.5).unwrap();
    vec![
        F::ChebyshevT,
        F::ChebyshevU,
        F::Legendre,
        F::Jacobi {
            alpha: half,
            beta: half + R::ONE,
        },
        F::Gegenbauer {
            lambda: half + R::ONE,
        },
        F::Gegenbauer {
            lambda: -half / (R::TWO),
        },
        F::Laguerre,
        F::GeneralizedLaguerre { alpha: half },
        F::HermiteH,
        F::HermiteHe,
    ]
}
fn number<C: NumericScalar>(real: f64, imag: f64) -> C {
    C::from_components(
        C::Real::try_from_f64_lossy(real).unwrap(),
        C::Real::try_from_f64_lossy(imag).unwrap(),
    )
}
fn calculus<C: NumericScalar>(imag: f64) {
    let policy = PolynomialApproximationPolicy::default();
    let mono = ApproxPolynomial::monomial_with_options(
        vec![
            number::<C>(1.0, imag),
            number(2.0, -imag),
            number(-0.5, 0.0),
            number(0.25, imag),
        ],
        policy,
    )
    .unwrap();
    for family in families() {
        for normalization in [N::Classical, N::Monic, N::Orthonormal] {
            let spec = OrthogonalBasisSpec::new(
                family,
                normalization,
                C::Real::try_from_f64_exact(0.25).unwrap(),
                C::Real::try_from_f64_exact(1.5).unwrap(),
            )
            .unwrap();
            let conversion = mono.to_orthogonal(spec).unwrap();
            let p = conversion.polynomial();
            let d = p.derivative().unwrap();
            let integral = p
                .antiderivative(number(0.5, imag), number(2.0, -imag))
                .unwrap();
            assert_close(
                integral.evaluate(number(0.5, imag)).unwrap(),
                number(2.0, -imag),
                256.0,
            );
            let recovered = integral.derivative().unwrap();
            for point in [-0.8, 0.0, 0.3, 1.0] {
                let x = number(point, imag);
                let expected = mono.evaluate_with_derivative(x).unwrap();
                let actual = p.evaluate_with_derivative(x).unwrap();
                assert_close(actual.value(), expected.value(), 512.0);
                assert_close(actual.derivative(), expected.derivative(), 512.0);
                assert_close(d.evaluate(x).unwrap(), expected.derivative(), 512.0);
                assert_close(recovered.evaluate(x).unwrap(), expected.value(), 1024.0);
                assert_close(
                    p.to_monomial().unwrap().polynomial().evaluate(x).unwrap(),
                    expected.value(),
                    1024.0,
                );
            }
            let target =
                OrthogonalBasisSpec::new(F::Legendre, N::Orthonormal, C::Real::ZERO, C::Real::ONE)
                    .unwrap();
            let converted = p.to_orthogonal(target).unwrap();
            assert_close(
                converted.polynomial().evaluate(number(0.3, imag)).unwrap(),
                mono.evaluate(number(0.3, imag)).unwrap(),
                1024.0,
            );
        }
    }
}
#[test]
fn real32_calculus() {
    calculus::<f32>(0.0);
}
#[test]
fn real64_calculus() {
    calculus::<f64>(0.0);
}
#[test]
fn complex32_calculus() {
    calculus::<Complex<f32>>(0.25);
}
#[test]
fn complex64_calculus() {
    calculus::<Complex<f64>>(0.25);
}

#[test]
fn analytic_normalizations_and_integrals() {
    let policy = PolynomialApproximationPolicy::default();
    for family in families::<f64>() {
        let spec = OrthogonalBasisSpec::new(family, N::Monic, 2.0, 3.0).unwrap();
        let p = ApproxPolynomial::orthogonal_with_plan(
            vec![0.0_f64, 0.0, 0.0, 1.0],
            spec.prepare(3, Default::default()).unwrap(),
            policy,
        )
        .unwrap();
        assert!(
            (p.to_monomial().unwrap().polynomial().basis().coefficients()[3] - 1.0).abs() < 1e-14
        );
    }
    let spec = OrthogonalBasisSpec::new(F::Legendre, N::Orthonormal, 2.0, 3.0).unwrap();
    let constant = ApproxPolynomial::orthogonal_with_plan(
        vec![1.0],
        spec.prepare(0, Default::default()).unwrap(),
        policy,
    )
    .unwrap();
    assert_close(constant.evaluate(10.0).unwrap(), 1.0 / 6.0_f64.sqrt(), 8.0);
    let p = ApproxPolynomial::monomial_with_options(vec![1.0, 2.0, 3.0], policy)
        .unwrap()
        .to_orthogonal(spec)
        .unwrap();
    assert_close(p.polynomial().integral(-1.0, 2.0).unwrap(), 15.0, 256.0);
    assert_close(p.polynomial().integral(2.0, -1.0).unwrap(), -15.0, 256.0);
    assert_eq!(constant.derivative().unwrap().evaluate(0.0).unwrap(), 0.0);
    assert_eq!(constant.integral(1.0, 1.0).unwrap(), 0.0);
}

#[test]
fn high_degree_and_scaled_intermediate_range() {
    let spec = OrthogonalBasisSpec::new(F::HermiteH, N::Classical, 0.0, 1.0).unwrap();
    let mut coefficients = vec![0.0; 301];
    coefficients[300] = 1e-300;
    let basis_plan = spec
        .prepare(coefficients.len() - 1, Default::default())
        .unwrap();
    let p = ApproxPolynomial::orthogonal_with_plan(
        coefficients,
        basis_plan,
        PolynomialApproximationPolicy::default(),
    )
    .unwrap();
    // H_300(0)=300!/150!, with positive sign, computed in log space independently.
    let expected =
        ((151..=300).map(|n| (n as f64).ln()).sum::<f64>() - 300.0 * 10.0_f64.ln()).exp();
    assert_close(p.evaluate(0.0).unwrap(), expected, 2000.0);
    for normalization in [N::Classical, N::Monic, N::Orthonormal] {
        let spec = OrthogonalBasisSpec::new(F::Legendre, normalization, 0.0, 1.0).unwrap();
        let plan = spec
            .prepare(256, scioxide_poly::PolynomialExecutionOptions::new())
            .unwrap();
        let mut coefficients = vec![0.0; 257];
        coefficients[256] = 1.0;
        let at_one = plan.evaluate_with_derivative(&coefficients, 1.0).unwrap();
        let at_minus = plan.evaluate_with_derivative(&coefficients, -1.0).unwrap();
        assert_close(at_one.value(), at_minus.value(), 4096.0);
        assert_close(at_one.derivative(), -at_minus.derivative(), 4096.0);
        if normalization == N::Classical {
            assert_close(at_one.value(), 1.0, 65536.0);
            assert_close(at_one.derivative(), 32896.0, 65536.0);
        }
    }
}

#[test]
fn borrowed_operator_and_batch_failures_are_transactional() {
    let spec = OrthogonalBasisSpec::new(
        F::Jacobi {
            alpha: 0.25,
            beta: 0.75,
        },
        N::Classical,
        0.0,
        1.0,
    )
    .unwrap();
    let plan = spec
        .prepare(3, scioxide_poly::PolynomialExecutionOptions::new())
        .unwrap();
    let coefficients = [1.0, 0.5, -0.1, 0.01];
    let operator = plan.derivative_operator();
    let mut output = [99.0; 3];
    let mut scratch = [0.0; 3];
    assert_eq!(
        scioxide_core::LinearOperator::<f64>::apply_workspace_len(&operator).unwrap(),
        3
    );
    assert!(matches!(
        operator
            .apply_into(&coefficients, &mut output, &mut scratch[..2])
            .unwrap_err()
            .view(),
        scioxide_core::SciOxideErrorView::DimensionMismatch { .. }
    ));
    assert_eq!(output, [99.0; 3]);
    let mut invalid = coefficients;
    invalid[2] = f64::NAN;
    assert!(
        operator
            .apply_into(&invalid, &mut output, &mut scratch)
            .is_err()
    );
    assert_eq!(output, [99.0; 3]);
    operator
        .apply_into(&coefficients, &mut output, &mut scratch)
        .unwrap();
    assert_eq!(output.as_slice(), operator.apply(&coefficients).unwrap());
    let points = [-0.5, 0.0, 0.5];
    plan.evaluate_batch_into(
        &coefficients,
        &points,
        &mut output,
        &mut scratch,
        ExecutionContext::sequential(),
    )
    .unwrap();
    for (point, actual) in points.into_iter().zip(output) {
        assert_close(
            actual,
            plan.evaluate_with_derivative(&coefficients, point)
                .unwrap()
                .value(),
            1.0,
        );
    }
    let before = output;
    assert!(
        plan.evaluate_batch_into(
            &coefficients,
            &[0.0, f64::NAN, 1.0],
            &mut output,
            &mut scratch,
            ExecutionContext::sequential()
        )
        .is_err()
    );
    assert_eq!(output, before);
    assert!(plan.evaluate_with_derivative::<f64>(&[], 0.0).is_err());
}

#[test]
fn gegenbauer_and_domain_contracts() {
    assert!(matches!(
        (F::Gegenbauer { lambda: 0.0_f64 }.recurrence_coefficients(0))
            .unwrap_err()
            .view(),
        scioxide_core::SciOxideErrorView::DomainError { .. }
    ));
    assert!(
        F::Gegenbauer { lambda: -0.5_f64 }
            .recurrence_coefficients(0)
            .is_err()
    );
    assert!(
        F::Gegenbauer { lambda: f64::NAN }
            .recurrence_coefficients(0)
            .is_err()
    );
    let g = F::Gegenbauer { lambda: 0.5_f64 };
    for n in 0..20 {
        assert_close(
            g.evaluate(n, 0.3).unwrap(),
            F::Legendre.evaluate(n, 0.3).unwrap(),
            128.0,
        );
    }
    for family in families::<f64>() {
        assert!(OrthogonalBasisSpec::new(family, N::Classical, 0.0, 0.0).is_err());
        assert!(OrthogonalBasisSpec::new(family, N::Classical, f64::INFINITY, 1.0).is_err());
    }
}

fn references<C: NumericScalar>() {
    for line in include_str!("data/orthogonal.csv").lines().skip(1) {
        let fields: Vec<_> = line.split(',').collect();
        let degree: usize = fields[1].parse().unwrap();
        // Binary32 coverage is through degree 32 (classical Hermite
        // degree 64 exceeds its output range). Binary64 covers degree 64.
        if C::Real::EPSILON > C::Real::try_from_f64_lossy(1e-10).unwrap() && degree > 32 {
            continue;
        }
        let family = match fields[0] {
            "legendre" => F::Legendre,
            "chebyshev_t" => F::ChebyshevT,
            "chebyshev_u" => F::ChebyshevU,
            "jacobi" => F::Jacobi {
                alpha: C::Real::try_from_f64_exact(0.25).unwrap(),
                beta: C::Real::try_from_f64_exact(0.75).unwrap(),
            },
            "gegenbauer" => F::Gegenbauer {
                lambda: C::Real::try_from_f64_exact(1.5).unwrap(),
            },
            "gegenbauer_negative" => F::Gegenbauer {
                lambda: C::Real::try_from_f64_exact(-0.25).unwrap(),
            },
            "laguerre" => F::Laguerre,
            "generalized_laguerre" => F::GeneralizedLaguerre {
                alpha: C::Real::try_from_f64_exact(0.5).unwrap(),
            },
            "hermite_h" => F::HermiteH,
            _ => F::HermiteHe,
        };
        let spec =
            OrthogonalBasisSpec::new(family, N::Classical, C::Real::ZERO, C::Real::ONE).unwrap();
        let mut coefficients = vec![C::ZERO; degree + 1];
        coefficients[degree] = C::ONE;
        let basis_plan = spec
            .prepare(coefficients.len() - 1, Default::default())
            .unwrap();
        let p = ApproxPolynomial::orthogonal_with_plan(
            coefficients,
            basis_plan,
            PolynomialApproximationPolicy::default(),
        )
        .unwrap();
        let x = number::<C>(fields[2].parse().unwrap(), 0.0);
        let evaluation = p.evaluate_with_derivative(x).unwrap();
        assert_close(
            evaluation.value(),
            number(fields[3].parse().unwrap(), 0.0),
            64.0 * (degree + 1) as f64,
        );
        assert_close(
            evaluation.derivative(),
            number(fields[4].parse().unwrap(), 0.0),
            64.0 * (degree + 1) as f64,
        );
        assert_close(
            C::from_real(family.evaluate(degree, x.real_component()).unwrap()),
            number(fields[3].parse().unwrap(), 0.0),
            64.0 * (degree + 1) as f64,
        );
        assert_close(
            C::from_real(
                family
                    .evaluate_derivative(degree, x.real_component())
                    .unwrap(),
            ),
            number(fields[4].parse().unwrap(), 0.0),
            64.0 * (degree + 1) as f64,
        );
    }
}
#[test]
fn exact_rational_reference_f64() {
    references::<f64>();
}
#[test]
fn exact_rational_reference_f32() {
    references::<f32>();
}
#[test]
fn exact_rational_reference_complex64() {
    references::<Complex<f64>>();
}
#[test]
fn exact_rational_reference_complex32() {
    references::<Complex<f32>>();
}

#[cfg(feature = "std")]
#[test]
fn injected_executor_matches_sequential_bits() {
    use core::num::NonZeroUsize;
    use scioxide_core::ThreadExecutor;
    let pool = ThreadExecutor::new(NonZeroUsize::new(2).unwrap());
    let plan = OrthogonalBasisSpec::new(F::Legendre, N::Orthonormal, 0.0_f64, 1.0)
        .unwrap()
        .prepare(
            3,
            scioxide_poly::PolynomialExecutionOptions::new()
                .with_execution_policy(ExecutionPolicy::Parallel(NonZeroUsize::new(2).unwrap())),
        )
        .unwrap();
    let coefficients = [1.0, 0.2, 0.3, 0.4];
    let points: Vec<_> = (0..101).map(|n| n as f64 / 100.0).collect();
    let mut sequential = vec![0.0; points.len()];
    let mut parallel = sequential.clone();
    let mut scratch = sequential.clone();
    plan.evaluate_batch_into(
        &coefficients,
        &points,
        &mut sequential,
        &mut scratch,
        ExecutionContext::sequential(),
    )
    .unwrap();
    plan.evaluate_batch_into(
        &coefficients,
        &points,
        &mut parallel,
        &mut scratch,
        ExecutionContext::new(&pool),
    )
    .unwrap();
    assert_eq!(sequential, parallel);
}

#[test]
fn existing_bases_share_calculus_and_conversion() {
    let policy = PolynomialApproximationPolicy::default();
    let monomial =
        ApproxPolynomial::monomial_with_options(vec![1.0_f64, 2.0, 3.0], policy).unwrap();
    let chebyshev = monomial
        .to_orthogonal(
            scioxide_poly::OrthogonalBasisSpec::new(
                scioxide_poly::OrthogonalFamily::ChebyshevT,
                scioxide_poly::BasisNormalization::Classical,
                0.0,
                1.0,
            )
            .unwrap(),
        )
        .unwrap();
    let legendre = monomial
        .to_orthogonal(
            scioxide_poly::OrthogonalBasisSpec::new(
                scioxide_poly::OrthogonalFamily::Legendre,
                scioxide_poly::BasisNormalization::Classical,
                0.0,
                1.0,
            )
            .unwrap(),
        )
        .unwrap();
    let bernstein = monomial.to_bernstein(-2.0, 3.0).unwrap();
    let spec = OrthogonalBasisSpec::new(F::HermiteHe, N::Orthonormal, 0.25, 1.5).unwrap();
    let x = 0.375;
    let expected = monomial.evaluate_with_derivative(x).unwrap();
    assert_close(
        chebyshev
            .polynomial()
            .derivative()
            .unwrap()
            .evaluate(x)
            .unwrap(),
        expected.derivative(),
        64.0,
    );
    assert_close(
        legendre
            .polynomial()
            .derivative()
            .unwrap()
            .evaluate(x)
            .unwrap(),
        expected.derivative(),
        64.0,
    );
    assert_close(
        bernstein
            .polynomial()
            .derivative()
            .unwrap()
            .evaluate(x)
            .unwrap(),
        expected.derivative(),
        128.0,
    );
    assert_close(
        chebyshev
            .polynomial()
            .antiderivative(0.0, 2.0)
            .unwrap()
            .evaluate(0.0)
            .unwrap(),
        2.0,
        64.0,
    );
    assert_close(
        legendre
            .polynomial()
            .antiderivative(0.0, 2.0)
            .unwrap()
            .evaluate(0.0)
            .unwrap(),
        2.0,
        64.0,
    );
    assert_close(
        bernstein
            .polynomial()
            .antiderivative(0.0, 2.0)
            .unwrap()
            .evaluate(0.0)
            .unwrap(),
        2.0,
        128.0,
    );
    for converted in [
        chebyshev.polynomial().to_orthogonal(spec).unwrap(),
        legendre.polynomial().to_orthogonal(spec).unwrap(),
        bernstein.polynomial().to_orthogonal(spec).unwrap(),
    ] {
        assert_close(
            converted.polynomial().evaluate(x).unwrap(),
            expected.value(),
            256.0,
        );
    }
    assert_close(
        bernstein.polynomial().integral(-1.0, 2.0).unwrap(),
        15.0,
        512.0,
    );
}

#[test]
fn coefficient_connections_and_condition_estimates_at_degree32() {
    let spec = OrthogonalBasisSpec::new(F::ChebyshevT, N::Classical, 0.0, 1.0).unwrap();
    let coefficients: Vec<_> = (0..33).map(|k| 2.0_f64.powi(-k)).collect();
    let p = ApproxPolynomial::orthogonal_with_plan(
        coefficients.clone(),
        spec.prepare(coefficients.len() - 1, Default::default())
            .unwrap(),
        PolynomialApproximationPolicy::default(),
    )
    .unwrap();
    let target = OrthogonalBasisSpec::new(F::Legendre, N::Orthonormal, 0.0, 1.0).unwrap();
    let converted = p.to_orthogonal(target).unwrap();
    let back = converted.polynomial().to_orthogonal(spec).unwrap();
    let scale = coefficients.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
    let error = coefficients
        .iter()
        .zip(back.polynomial().basis().coefficients())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max)
        / scale;
    assert!(error < 128.0 * f64::EPSILON * (1.0 + converted.condition().condition_estimate()));
    let cancellation = ApproxPolynomial::orthogonal_with_plan(
        vec![1.0_f64, -1.0 + 1e-10],
        spec.prepare(1, Default::default()).unwrap(),
        PolynomialApproximationPolicy::default(),
    )
    .unwrap();
    assert!(
        cancellation
            .evaluate_with_derivative(1.0)
            .unwrap()
            .condition()
            .condition_estimate()
            > 1e9
    );
}

#[test]
fn scaled_retry_preserves_subnormal_coefficients() {
    let spec = OrthogonalBasisSpec::new(F::ChebyshevT, N::Classical, 0.0_f64, 1.0).unwrap();
    let p = ApproxPolynomial::orthogonal_with_plan(
        vec![0.0, f64::from_bits(1)],
        spec.prepare(1, Default::default()).unwrap(),
        PolynomialApproximationPolicy::default(),
    )
    .unwrap();
    let evaluation = p.evaluate_with_derivative(2.0).unwrap();
    assert_eq!(evaluation.value().to_bits(), 2);
    assert_eq!(evaluation.derivative().to_bits(), 1);
}

#[test]
fn bernstein_trailing_zeros_preserve_degree_and_calculus() {
    let policy = PolynomialApproximationPolicy::default();
    let p = ApproxPolynomial::bernstein_with_options(vec![1.0_f64, 0.0], 0.0, 1.0, policy).unwrap();
    assert_eq!(p.degree(), 1);
    assert_eq!(p.evaluate(0.5).unwrap(), 0.5);
    let p = ApproxPolynomial::bernstein_with_options(vec![0.0_f64, 1.0, 1.0], 0.0, 1.0, policy)
        .unwrap();
    let derivative = p.derivative().unwrap();
    assert_eq!(derivative.degree(), 1);
    assert_eq!(derivative.basis().coefficients(), &[2.0, 0.0]);
    assert_eq!(derivative.evaluate(0.5).unwrap(), 1.0);
    assert_close(
        p.antiderivative(0.0, 0.0)
            .unwrap()
            .derivative()
            .unwrap()
            .evaluate(0.75)
            .unwrap(),
        p.evaluate(0.75).unwrap(),
        64.0,
    );
}

#[test]
fn empty_batches_and_cancellation_preserve_output() {
    let plan = OrthogonalBasisSpec::new(F::Legendre, N::Classical, 0.0_f64, 1.0)
        .unwrap()
        .prepare(1, scioxide_poly::PolynomialExecutionOptions::new())
        .unwrap();
    plan.evaluate_batch_into(
        &[1.0, 2.0],
        &[],
        &mut [],
        &mut [],
        ExecutionContext::sequential(),
    )
    .unwrap();
    let cancellation = scioxide_core::CancellationToken::new();
    cancellation.cancel();
    let mut output = [99.0];
    let mut scratch = [0.0];
    assert!(
        plan.evaluate_batch_into(
            &[1.0, 2.0],
            &[0.0],
            &mut output,
            &mut scratch,
            ExecutionContext::sequential().with_cancellation(&cancellation)
        )
        .is_err()
    );
    assert_eq!(output, [99.0]);
}

#[test]
fn bernstein_calculus_balances_unrepresentable_interval_width() {
    let p = ApproxPolynomial::bernstein(vec![0.0_f64, 1.0], -f64::MAX, f64::MAX).unwrap();
    let evaluation = p.evaluate_with_derivative(0.0).unwrap();
    assert_eq!(evaluation.value(), 0.5);
    assert!(evaluation.derivative() > 0.0);
    assert_eq!(
        evaluation.derivative(),
        p.derivative().unwrap().evaluate(0.0).unwrap()
    );
    let p = ApproxPolynomial::bernstein(vec![-f64::MAX, f64::MAX], -f64::MAX, f64::MAX).unwrap();
    assert_eq!(p.evaluate_with_derivative(0.0).unwrap().derivative(), 1.0);
}

#[test]
fn gegenbauer_small_parameters_retain_classical_and_normalized_limits() {
    for lambda in [
        1e-12_f64,
        -1e-12,
        1e-300,
        -1e-300,
        f64::from_bits(1),
        -f64::from_bits(1),
    ] {
        let family = F::Gegenbauer { lambda };
        assert_eq!(family.evaluate(2, 0.0).unwrap(), -lambda);
        let spec =
            OrthogonalBasisSpec::new(F::Gegenbauer { lambda }, N::Orthonormal, 0.0, 1.0).unwrap();
        let p = ApproxPolynomial::orthogonal_with_plan(
            vec![0.0_f64, 0.0, 1.0],
            spec.prepare(2, Default::default()).unwrap(),
            PolynomialApproximationPolicy::default(),
        )
        .unwrap();
        let expected = -(2.0 / core::f64::consts::PI).sqrt() * lambda.signum();
        assert!((p.evaluate(0.0).unwrap() - expected).abs() < 4e-12);
    }
    for lambda in [1e-10_f32, -1e-10, 1e-30, -1e-30, f32::from_bits(1)] {
        assert_eq!(F::Gegenbauer { lambda }.evaluate(2, 0.0).unwrap(), -lambda);
    }
}

#[test]
fn gegenbauer_weight_accepts_parameters_adjacent_to_lower_boundary() {
    fn check<R: RealScalar>() {
        let half = R::HALF;
        let lambda = (-half).next_after(R::ZERO);
        let family = F::Gegenbauer { lambda };
        assert_eq!(family.weight(R::ZERO).unwrap(), R::ONE);
        let expected = R::try_from_f64_lossy(4.0 / 3.0).unwrap();
        for x in [-half, half] {
            let actual = family.weight(x).unwrap();
            assert!((actual - expected).abs() <= R::try_from_f64_exact(4.0).unwrap() * R::EPSILON);
        }
        assert!(matches!(
            (family.weight(R::ONE)).unwrap_err().view(),
            scioxide_core::SciOxideErrorView::DomainError { .. }
        ));
        assert!(matches!(
            (family.weight(R::NAN)).unwrap_err().view(),
            scioxide_core::SciOxideErrorView::NonFinite { .. }
        ));
    }
    check::<f32>();
    check::<f64>();
}

fn many_points<C: NumericScalar>(imaginary: f64) {
    for family in families() {
        for normalization in [N::Classical, N::Monic, N::Orthonormal] {
            let spec = OrthogonalBasisSpec::new(family, normalization, C::Real::ZERO, C::Real::ONE)
                .unwrap();
            let polynomial = ApproxPolynomial::orthogonal_with_plan(
                vec![
                    number::<C>(1.0, imaginary),
                    number(-0.5, 0.0),
                    number(0.25, -imaginary),
                    number(0.125, imaginary),
                ],
                spec.prepare(3, Default::default()).unwrap(),
                Default::default(),
            )
            .unwrap();
            let points: [C; 17] =
                core::array::from_fn(|i| number::<C>(i as f64 / 16.0 - 0.5, imaginary));
            for len in 0..=points.len() {
                let results: Vec<_> = polynomial
                    .evaluate_many_with_derivative(&points[..len])
                    .collect();
                assert_eq!(results.len(), len);
                for (result, &point) in results.into_iter().zip(&points[..len]) {
                    let actual = result.unwrap();
                    let expected = polynomial.evaluate_with_derivative(point).unwrap();
                    assert_close(actual.value(), expected.value(), 256.0);
                    assert_close(actual.derivative(), expected.derivative(), 256.0);
                    assert_close(
                        C::from_real(actual.condition().condition_estimate()),
                        C::from_real(expected.condition().condition_estimate()),
                        256.0,
                    );
                }
            }
        }
    }
}

#[test]
fn independent_points_keep_complete_reports_and_tail_order_in_all_widths() {
    many_points::<f32>(0.0);
    many_points::<f64>(0.0);
    many_points::<Complex<f32>>(0.125);
    many_points::<Complex<f64>>(0.125);
}

#[test]
fn independent_points_continue_after_input_errors_and_retry_extended_range() {
    let spec = OrthogonalBasisSpec::new(F::ChebyshevT, N::Classical, 0.0, 1.0).unwrap();
    let polynomial = ApproxPolynomial::orthogonal_with_plan(
        vec![1.0_f64, 0.5, -0.25],
        spec.prepare(2, Default::default()).unwrap(),
        Default::default(),
    )
    .unwrap();
    let points = [0.0, f64::NAN, 0.25, f64::INFINITY, 0.5];
    for (index, result) in polynomial
        .evaluate_many_with_derivative(&points)
        .enumerate()
    {
        if index == 1 || index == 3 {
            assert!(matches!(
                result.unwrap_err().view(),
                scioxide_core::SciOxideErrorView::NonFinite { .. }
            ));
        } else {
            assert_close(
                result.unwrap().value(),
                polynomial.evaluate(points[index]).unwrap(),
                64.0,
            );
        }
    }
    let spec = OrthogonalBasisSpec::new(F::HermiteH, N::Classical, 0.0, 1.0).unwrap();
    let mut coefficients = vec![0.0_f64; 301];
    coefficients[300] = 1e-300;
    let basis_plan = spec
        .prepare(coefficients.len() - 1, Default::default())
        .unwrap();
    let polynomial =
        ApproxPolynomial::orthogonal_with_plan(coefficients, basis_plan, Default::default())
            .unwrap();
    for (result, point) in polynomial
        .evaluate_many_with_derivative(&[0.0, 0.25, 0.5, 0.75, 0.0])
        .zip([0.0, 0.25, 0.5, 0.75, 0.0])
    {
        let expected = polynomial.evaluate_with_derivative(point).unwrap();
        let actual = result.unwrap();
        assert_close(actual.value(), expected.value(), 256.0);
        assert_close(actual.derivative(), expected.derivative(), 256.0);
    }
}

#[test]
fn independent_points_share_the_long_recurrence_and_affine_normalization() {
    for family in [F::ChebyshevT, F::Legendre, F::Laguerre] {
        for normalization in [N::Classical, N::Monic, N::Orthonormal] {
            let spec = OrthogonalBasisSpec::new(family, normalization, 0.25, 1.5).unwrap();
            let mut coefficients = vec![0.0_f64; 33];
            coefficients[0] = 0.5;
            coefficients[16] = -0.25;
            coefficients[32] = 0.125;
            let basis_plan = spec
                .prepare(coefficients.len() - 1, Default::default())
                .unwrap();
            let polynomial = ApproxPolynomial::orthogonal_with_plan(
                coefficients,
                basis_plan,
                Default::default(),
            )
            .unwrap();
            let points: [f64; 17] = core::array::from_fn(|i| i as f64 / 16.0 - 0.5);
            for (result, point) in polynomial
                .evaluate_many_with_derivative(&points)
                .zip(points)
            {
                let actual = result.unwrap();
                let expected = polynomial.evaluate_with_derivative(point).unwrap();
                assert_close(actual.value(), expected.value(), 512.0);
                assert_close(actual.derivative(), expected.derivative(), 512.0);
            }
        }
    }
}

#[test]
fn value_batches_do_not_project_an_unrequested_derivative() {
    let spec = OrthogonalBasisSpec::new(F::HermiteH, N::Classical, 0.0_f64, 1.0).unwrap();
    let plan = spec
        .prepare(301, scioxide_poly::PolynomialExecutionOptions::new())
        .unwrap();
    let mut coefficients = vec![0.0; 302];
    coefficients[301] = 1.0;
    let mut output = [1.0];
    plan.evaluate_batch_into(
        &coefficients,
        &[0.0],
        &mut output,
        &mut [0.0],
        ExecutionContext::sequential(),
    )
    .unwrap();
    assert_eq!(output, [0.0]);
}

#[test]
fn classical_chebyshev_lanes_match_independent_trigonometric_values() {
    let spec = OrthogonalBasisSpec::new(F::ChebyshevT, N::Classical, 0.0_f64, 1.0).unwrap();
    let mut coefficients = vec![0.0; 64];
    coefficients[63] = 1.0;
    let basis_plan = spec
        .prepare(coefficients.len() - 1, Default::default())
        .unwrap();
    let polynomial =
        ApproxPolynomial::orthogonal_with_plan(coefficients, basis_plan, Default::default())
            .unwrap();
    let angles: [f64; 17] = core::array::from_fn(|i| 0.125 + i as f64 / 16.0);
    let points = angles.map(f64::cos);
    for (result, angle) in polynomial
        .evaluate_many_with_derivative(&points)
        .zip(angles)
    {
        let evaluation = result.unwrap();
        assert_close(evaluation.value(), (63.0 * angle).cos(), 4096.0);
        assert_close(
            evaluation.derivative(),
            63.0 * (63.0 * angle).sin() / angle.sin(),
            4096.0,
        );
    }
}

#[test]
fn linear_calculus_matches_high_degree_closed_forms() {
    for family in [F::ChebyshevT, F::Legendre] {
        let spec = OrthogonalBasisSpec::new(family, N::Classical, 0.0, 1.0).unwrap();
        let mut coefficients = vec![0.0_f64; 1025];
        coefficients[1024] = 1.0;
        let basis_plan = spec
            .prepare(coefficients.len() - 1, Default::default())
            .unwrap();
        let polynomial =
            ApproxPolynomial::orthogonal_with_plan(coefficients, basis_plan, Default::default())
                .unwrap();
        let derivative = polynomial.derivative().unwrap();
        for (k, &actual) in derivative.basis().coefficients().iter().enumerate() {
            let expected = if k % 2 == 0 {
                0.0
            } else if family == F::ChebyshevT {
                2048.0
            } else {
                (2 * k + 1) as f64
            };
            assert_close(actual, expected, 512.0);
        }
        let primitive = polynomial.antiderivative(0.0, 0.0).unwrap();
        let (upper, lower) = if family == F::ChebyshevT {
            (1.0 / 2050.0, -1.0 / 2046.0)
        } else {
            (1.0 / 2049.0, -1.0 / 2049.0)
        };
        for (k, &actual) in primitive.basis().coefficients().iter().enumerate() {
            let expected = match k {
                1025 => upper,
                1023 => lower,
                _ => 0.0,
            };
            assert_close(actual, expected, 64.0);
        }
    }
}

#[test]
fn calculus_retries_extended_range_and_projects_only_final_coefficients() {
    let spec = OrthogonalBasisSpec::new(F::ChebyshevT, N::Classical, 0.0, 2.0).unwrap();
    let p = ApproxPolynomial::orthogonal_with_plan(
        vec![0.0, f64::MAX * 0.75, 0.0, f64::MAX * 0.125],
        spec.prepare(3, Default::default()).unwrap(),
        Default::default(),
    )
    .unwrap();
    let derivative = p.derivative().unwrap();
    assert_close(
        derivative.basis().coefficients()[0] / f64::MAX,
        0.5625,
        16.0,
    );
    assert_close(derivative.basis().coefficients()[2] / f64::MAX, 0.375, 16.0);
    let p = ApproxPolynomial::orthogonal_with_plan(
        vec![f64::MAX * 0.625, 0.0, f64::MAX * 0.25],
        spec.prepare(2, Default::default()).unwrap(),
        Default::default(),
    )
    .unwrap();
    let primitive = p.antiderivative(0.0, 0.0).unwrap();
    assert_close(primitive.basis().coefficients()[1] / f64::MAX, 1.0, 16.0);
    let spec = OrthogonalBasisSpec::new(F::Legendre, N::Orthonormal, 0.0, 1e-200).unwrap();
    let p = ApproxPolynomial::orthogonal_with_plan(
        vec![1e100],
        spec.prepare(0, Default::default()).unwrap(),
        Default::default(),
    )
    .unwrap();
    // The unanchored value is ~7e349, but its constant coordinate is finite.
    let primitive = p.antiderivative(1e150, 0.0).unwrap();
    assert_close(primitive.basis().coefficients()[0] / 1e250, -1.0, 2048.0);
    assert_close(
        primitive.derivative().unwrap().basis().coefficients()[0] / 1e100,
        1.0,
        2048.0,
    );
    let p = ApproxPolynomial::orthogonal_with_plan(
        vec![1e250],
        spec.prepare(0, Default::default()).unwrap(),
        Default::default(),
    )
    .unwrap();
    // The integral is finite although the unrequested endpoint derivative is not.
    assert_close(
        p.integral(0.0, 1e-200).unwrap() / 1e150,
        1.0 / 2.0_f64.sqrt(),
        2048.0,
    );
    for tiny in [f64::from_bits(1), -f64::from_bits(1)] {
        let spec = OrthogonalBasisSpec::new(F::ChebyshevT, N::Classical, 0.0, 1.0).unwrap();
        let p = ApproxPolynomial::orthogonal_with_plan(
            vec![0.0, tiny],
            spec.prepare(1, Default::default()).unwrap(),
            Default::default(),
        )
        .unwrap();
        assert_eq!(p.derivative().unwrap().basis().coefficients()[0], tiny);
        let primitive = ApproxPolynomial::orthogonal_with_plan(
            vec![tiny],
            spec.prepare(0, Default::default()).unwrap(),
            Default::default(),
        )
        .unwrap()
        .antiderivative(0.0, 0.0)
        .unwrap();
        assert_eq!(primitive.basis().coefficients()[1], tiny);
    }
}

fn prepared_conversions<C: NumericScalar>(imaginary: f64) {
    let input = vec![
        number::<C>(1.0, imaginary),
        number(-0.5, 0.0),
        number(0.25, -imaginary),
        number(0.125, imaginary),
    ];
    let mono = ApproxPolynomial::monomial(input).unwrap();
    for family in families() {
        for normalization in [N::Classical, N::Monic, N::Orthonormal] {
            let spec = OrthogonalBasisSpec::new(
                family,
                normalization,
                C::Real::try_from_f64_exact(0.25).unwrap(),
                C::Real::try_from_f64_exact(1.5).unwrap(),
            )
            .unwrap();
            let plan = spec.prepare(3, Default::default()).unwrap();
            let operator = plan.from_monomial_operator().unwrap();
            for _ in 0..2 {
                let cached = mono.to_orthogonal_with_operator(&operator).unwrap();
                let direct = mono.to_orthogonal(spec).unwrap();
                for (&actual, &expected) in cached
                    .polynomial()
                    .basis()
                    .coefficients()
                    .iter()
                    .zip(direct.polynomial().basis().coefficients())
                {
                    assert_close(actual, expected, 1024.0);
                }
                assert_close(
                    C::from_real(cached.condition().condition_estimate()),
                    C::from_real(direct.condition().condition_estimate()),
                    1024.0,
                );
                assert_close(
                    C::from_real(cached.condition().backward_error()),
                    C::from_real(direct.condition().backward_error()),
                    1024.0,
                );
                let polynomial = cached.polynomial();
                let to_power = polynomial.basis().plan().monomial_operator().unwrap();
                let back = polynomial.to_monomial_with_operator(&to_power).unwrap();
                for (&actual, &expected) in back
                    .polynomial()
                    .basis()
                    .coefficients()
                    .iter()
                    .zip(mono.basis().coefficients())
                {
                    assert_close(actual, expected, 2048.0);
                }
                let target = OrthogonalBasisSpec::new(
                    F::Legendre,
                    N::Classical,
                    C::Real::ZERO,
                    C::Real::ONE,
                )
                .unwrap()
                .prepare(6, Default::default())
                .unwrap();
                let connection = polynomial
                    .basis()
                    .plan()
                    .conversion_operator(&target)
                    .unwrap();
                let converted = polynomial.to_orthogonal_with_operator(&connection).unwrap();
                assert_close(
                    converted
                        .polynomial()
                        .evaluate(number(0.3, imaginary))
                        .unwrap(),
                    mono.evaluate(number(0.3, imaginary)).unwrap(),
                    2048.0,
                );
                assert!(polynomial.to_orthogonal_with_operator(&operator).is_err());
                assert!(polynomial.to_orthogonal_with_operator(&to_power).is_err());
                assert!(polynomial.to_monomial_with_operator(&connection).is_err());
                assert!(mono.to_orthogonal_with_operator(&connection).is_err());
            }
        }
    }
}
#[test]
fn immutable_conversion_operators_reuse_every_family_and_width() {
    prepared_conversions::<f32>(0.0);
    prepared_conversions::<f64>(0.0);
    prepared_conversions::<Complex<f32>>(0.125);
    prepared_conversions::<Complex<f64>>(0.125);
}

#[test]
fn borrowed_calculus_and_conversion_errors_preserve_output() {
    let spec = OrthogonalBasisSpec::new(F::ChebyshevT, N::Classical, 0.0, 1.0).unwrap();
    let plan = spec.prepare(2, Default::default()).unwrap();
    let derivative = plan.derivative_operator();
    let primitive = plan.antiderivative_plan().unwrap();
    let conversion = plan.monomial_operator().unwrap();
    for input in [
        [0.0, f64::NAN, 1.0],
        [0.0, f64::INFINITY, 1.0],
        [0.0, 0.0, f64::MAX],
    ] {
        let mut output = [99.0; 2];
        assert!(
            derivative
                .apply_into(&input, &mut output, &mut [0.0; 2])
                .is_err()
        );
        assert_eq!(output, [99.0; 2]);
    }
    let mut output = [99.0; 4];
    for boundary in [(f64::NAN, 0.0), (0.0, f64::INFINITY)] {
        assert!(
            primitive
                .apply_into(
                    &[1.0, 2.0, 3.0],
                    boundary,
                    &mut output,
                    &mut [0.0; 4],
                    scioxide_core::ExecutionContext::sequential()
                )
                .is_err()
        );
        assert_eq!(output, [99.0; 4]);
    }
    let mut converted = [99.0; 3];
    assert!(
        conversion
            .apply_into(&[0.0, 0.0, f64::MAX], &mut converted, &mut [0.0; 3])
            .is_err()
    );
    assert_eq!(converted, [99.0; 3]);
    assert!(
        conversion
            .apply_into(&[1.0, 2.0, 3.0], &mut converted, &mut [0.0; 2])
            .is_err()
    );
    assert_eq!(converted, [99.0; 3]);
    let wrong_degree = ApproxPolynomial::monomial(vec![1.0_f64, 2.0]).unwrap();
    assert!(
        wrong_degree
            .to_orthogonal_with_operator(&plan.from_monomial_operator().unwrap())
            .is_err()
    );
    let negative = ApproxPolynomial::orthogonal_with_plan(
        vec![0.0, 1.0, 0.0],
        spec.prepare(2, Default::default()).unwrap(),
        Default::default(),
    )
    .unwrap();
    assert_eq!(
        negative
            .derivative()
            .unwrap()
            .derivative()
            .unwrap()
            .derivative()
            .unwrap()
            .basis()
            .coefficients(),
        &[0.0]
    );
    assert_close(
        negative.integral(1.0, -1.0).unwrap(),
        -negative.integral(-1.0, 1.0).unwrap(),
        16.0,
    );
}

fn subnormal_calculus_and_conversion<C: NumericScalar>() {
    let tiny = C::from_components(
        C::Real::MIN_SUBNORMAL,
        if C::IS_REAL {
            C::Real::ZERO
        } else {
            -C::Real::MIN_SUBNORMAL
        },
    );
    let spec =
        OrthogonalBasisSpec::new(F::ChebyshevT, N::Classical, C::Real::ZERO, C::Real::ONE).unwrap();
    let polynomial = ApproxPolynomial::orthogonal_with_plan(
        vec![C::ZERO, tiny, C::ZERO],
        spec.prepare(2, Default::default()).unwrap(),
        Default::default(),
    )
    .unwrap();
    assert_eq!(
        polynomial.derivative().unwrap().basis().coefficients()[0],
        tiny
    );
    let constant = ApproxPolynomial::orthogonal_with_plan(
        vec![tiny],
        spec.prepare(0, Default::default()).unwrap(),
        Default::default(),
    )
    .unwrap();
    assert_eq!(
        constant
            .antiderivative(C::ZERO, C::ZERO)
            .unwrap()
            .basis()
            .coefficients()[1],
        tiny
    );
    let power = constant.to_monomial().unwrap();
    assert_eq!(power.polynomial().basis().coefficients(), &[tiny]);
    assert_eq!(
        power
            .polynomial()
            .to_orthogonal(spec)
            .unwrap()
            .polynomial()
            .basis()
            .coefficients(),
        &[tiny]
    );
    let identical = polynomial.to_orthogonal(spec).unwrap();
    assert_eq!(
        identical.polynomial().basis().coefficients(),
        polynomial.basis().coefficients()
    );
    assert_eq!(identical.polynomial().degree(), 2);
}
#[test]
fn scaled_calculus_and_conversions_preserve_subnormal_components_in_every_width() {
    subnormal_calculus_and_conversion::<f32>();
    subnormal_calculus_and_conversion::<f64>();
    subnormal_calculus_and_conversion::<Complex<f32>>();
    subnormal_calculus_and_conversion::<Complex<f64>>();
}

#[test]
fn calculus_covers_degenerate_gegenbauer_parameter_limits() {
    for lambda in [
        1e-300,
        -1e-300,
        f64::MIN_POSITIVE,
        -f64::MIN_POSITIVE,
        f64::from_bits(1),
        -f64::from_bits(1),
        (-0.5_f64).next_after(0.0),
    ] {
        for normalization in [N::Classical, N::Monic, N::Orthonormal] {
            let spec = OrthogonalBasisSpec::new(F::Gegenbauer { lambda }, normalization, 0.25, 1.5)
                .unwrap();
            let polynomial = ApproxPolynomial::orthogonal_with_plan(
                vec![lambda, 0.25, 0.125],
                spec.prepare(2, Default::default()).unwrap(),
                Default::default(),
            )
            .unwrap();
            let derivative = polynomial.derivative().unwrap();
            let primitive = polynomial.antiderivative(0.25, 0.0).unwrap();
            for point in [0.0, 0.25, 0.5] {
                assert_close(
                    derivative.evaluate(point).unwrap(),
                    polynomial
                        .evaluate_with_derivative(point)
                        .unwrap()
                        .derivative(),
                    4096.0,
                );
                assert_close(
                    primitive.derivative().unwrap().evaluate(point).unwrap(),
                    polynomial.evaluate(point).unwrap(),
                    4096.0,
                );
            }
        }
    }
}
