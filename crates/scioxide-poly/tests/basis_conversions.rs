//! Exact rational expansions generated independently of the conversion recurrences.
#![allow(clippy::unwrap_used)]
use scioxide_core::*;
use scioxide_poly::*;

fn check_references<C: NumericScalar>() {
    let r = |x: f64| C::Real::try_from_f64_lossy(x).unwrap();
    let scalar = |x: f64| C::from_components(r(x), r(x / 2.));
    for line in include_str!("data/conversions.csv").lines().skip(1) {
        let fields: Vec<_> = line.split(',').collect();
        let degree: usize = fields[2].parse().unwrap();
        let lower = r(fields[3].parse().unwrap());
        let upper = r(fields[4].parse().unwrap());
        let parse = |text: &str| {
            text.split('|')
                .map(|s| s.parse::<f64>().unwrap())
                .collect::<Vec<_>>()
        };
        let source = parse(fields[5]).into_iter().map(scalar).collect();
        let actual = match fields[1] {
            "b_to_m" => ApproxPolynomial::bernstein(source, lower, upper)
                .unwrap()
                .to_monomial()
                .unwrap()
                .into_polynomial()
                .basis()
                .coefficients()
                .to_vec(),
            "m_to_b" => ApproxPolynomial::monomial(source)
                .unwrap()
                .to_bernstein(lower, upper)
                .unwrap()
                .into_polynomial()
                .basis()
                .coefficients()
                .to_vec(),
            "n_to_m" => {
                let nodes = if degree == 0 {
                    vec![]
                } else {
                    parse(fields[6])
                        .into_iter()
                        .map(|x| C::from_real(r(x)))
                        .collect()
                };
                ApproxPolynomial::newton(source, nodes)
                    .unwrap()
                    .to_monomial()
                    .unwrap()
                    .into_polynomial()
                    .basis()
                    .coefficients()
                    .to_vec()
            }
            _ => unreachable!(),
        };
        let expected = parse(fields[7]);
        let majorant = parse(fields[8]);
        for (index, (&reference, &bound)) in expected.iter().zip(&majorant).enumerate() {
            let value = actual.get(index).copied().unwrap_or(C::ZERO);
            let error = (value - scalar(reference)).abs();
            let tolerance = r(64. * (degree + 1) as f64) * C::Real::EPSILON * r(bound)
                + r(4. * (degree + 1) as f64) * C::Real::MIN_SUBNORMAL;
            assert!(
                error <= tolerance,
                "{} {}[{index}]: error {error:?}, tolerance {tolerance:?}",
                fields[0],
                fields[1]
            );
        }
    }
}

#[test]
fn rational_expansions_through_degree_128_real_and_complex() {
    // Real types discard imaginary components in from_components.
    check_references::<f32>();
    check_references::<f64>();
    check_references::<Complex<f32>>();
    check_references::<Complex<f64>>();
}

#[test]
fn constant_control_nets_do_not_acquire_spurious_coefficients() {
    for degree in [0, 1, 28, 57, 68, 128] {
        let p = ApproxPolynomial::bernstein(vec![1.0_f64; degree + 1], 0., 1.).unwrap();
        assert_eq!(
            p.to_monomial().unwrap().polynomial().basis().coefficients(),
            &[1.]
        );
        let p = ApproxPolynomial::bernstein(vec![1.0_f32; degree + 1], 0., 1.).unwrap();
        assert_eq!(
            p.to_monomial().unwrap().polynomial().basis().coefficients(),
            &[1.]
        );
    }
}

#[test]
fn wide_and_tiny_affine_maps_preserve_range_or_return_typed_errors() {
    let p = ApproxPolynomial::bernstein(vec![0.0_f64, 1.], -1e308, 1e308).unwrap();
    let converted = p.to_monomial().unwrap();
    let coefficients = converted.polynomial().basis().coefficients();
    assert!((coefficients[0] - 0.5).abs() <= f64::EPSILON);
    assert!((coefficients[1] - 5e-309).abs() <= f64::from_bits(1));
    let back = converted.polynomial().to_bernstein(-1e308, 1e308).unwrap();
    assert!(back.polynomial().basis().coefficients()[0].abs() <= 4. * f64::EPSILON);
    assert!((back.polynomial().basis().coefficients()[1] - 1.).abs() <= 4. * f64::EPSILON);

    let p = ApproxPolynomial::bernstein(vec![0.0_f32, 1.], -3e38, 3e38).unwrap();
    let converted = p.to_monomial().unwrap();
    assert!((converted.polynomial().basis().coefficients()[0] - 0.5).abs() <= f32::EPSILON);
    assert!(
        (converted.polynomial().basis().coefficients()[1] - (1. / 6e38_f64) as f32).abs()
            <= f32::from_bits(1)
    );

    // Finite but unrepresentable final coefficients are numerical errors.
    let p = ApproxPolynomial::bernstein(vec![0.0_f64, f64::MAX], 0., f64::MIN_POSITIVE).unwrap();
    assert!(matches!(
        p.to_monomial().unwrap_err().view(),
        SciOxideErrorView::NumericalInstability {
            issue: NumericalIssue::Overflow,
            ..
        }
    ));
    let p = ApproxPolynomial::bernstein(vec![0.0_f64, f64::from_bits(1)], -1e308, 1e308).unwrap();
    assert!(matches!(
        p.to_monomial().unwrap_err().view(),
        SciOxideErrorView::NumericalInstability {
            issue: NumericalIssue::Underflow,
            ..
        }
    ));
}

#[test]
fn newton_retains_valid_zero_tails_and_rejects_invalid_layouts() {
    let p = ApproxPolynomial::newton(vec![1.0_f64, 2., 0.], vec![0., 1.]).unwrap();
    assert_eq!(p.degree(), 2);
    assert_eq!(p.basis().coefficients(), &[1., 2., 0.]);
    assert_eq!(p.basis().nodes(), &[0., 1.]);
    assert_eq!(p.classify_coefficients().unwrap().effective_degree(), 1);
    assert_eq!(
        p.to_monomial().unwrap().polynomial().basis().coefficients(),
        &[1., 2.]
    );
    let zero = ApproxPolynomial::newton(vec![0.0_f64; 3], vec![0., 1.]).unwrap();
    assert_eq!(
        zero.to_monomial()
            .unwrap()
            .polynomial()
            .basis()
            .coefficients(),
        &[0.]
    );
    assert!(ApproxPolynomial::newton(vec![1.0_f64, 2., 0.], vec![0.]).is_err());
    assert!(ApproxPolynomial::newton(Vec::<f64>::new(), vec![]).is_err());
    assert!(ApproxPolynomial::newton(vec![1.0_f64, f64::NAN], vec![0.]).is_err());
}

#[test]
fn newton_projects_after_intermediate_overflow_cancels() {
    fn check<R: RealScalar + NumericScalar<Real = R>>(m: R) {
        let p = ApproxPolynomial::newton(
            vec![R::ONE, -m, m],
            vec![R::ZERO, -R::try_from_usize_exact(2).unwrap()],
        )
        .unwrap();
        // 1 - M*x + M*x*(x+2) = 1 + M*x + M*x². The inner
        // product 2*M exceeds native range before subtracting M.
        let converted = p.to_monomial().unwrap();
        let coefficients = converted.polynomial().basis().coefficients();
        assert_eq!(coefficients[0], R::ONE);
        assert!((coefficients[1] / m - R::ONE).abs() <= R::EPSILON);
        assert!((coefficients[2] / m - R::ONE).abs() <= R::EPSILON);
    }
    check(1e308_f64);
    check(2e38_f32);
}
