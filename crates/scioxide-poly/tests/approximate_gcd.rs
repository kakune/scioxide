//! Independent high-precision cofactor least-squares references.
#![allow(clippy::unwrap_used)]
use scioxide_core::{RankThreshold, SciOxideErrorKind, Tolerance};
use scioxide_linalg::RankPolicy;
use scioxide_poly::PolynomialBasis;
use scioxide_poly::{ApproxPolynomial, PolynomialApproximationPolicy, approximate_gcd};

#[test]
fn moderate_degree_gcds_match_independent_rank_and_factor_references() {
    for line in include_str!("data/gcd.csv").lines().skip(1) {
        let fields: Vec<_> = line.split(',').collect();
        let degree: usize = fields[1].parse().unwrap();
        let relative: f64 = fields[4].parse().unwrap();
        let rank: usize = fields[5].parse().unwrap();
        let condition: f64 = fields[6].parse().unwrap();
        let fit_error: f64 = fields[7].parse().unwrap();
        let parse = |s: &str| {
            s.split('|')
                .map(|x| x.parse::<f64>().unwrap())
                .collect::<Vec<_>>()
        };
        let expected = parse(fields[8]);
        let left = parse(fields[9]);
        let right = parse(fields[10]);
        let policy = PolynomialApproximationPolicy::new(
            Tolerance::new(0., 1e-8).unwrap(),
            RankPolicy::Threshold(RankThreshold::new(0., relative).unwrap()),
        );
        for (left_scale, right_scale) in [(1., 1.), (2.0_f64.powi(400), 2.0_f64.powi(-400))] {
            let a = ApproxPolynomial::monomial_with_options(
                left.iter().map(|x| x * left_scale).collect(),
                policy,
            )
            .unwrap();
            let b = ApproxPolynomial::monomial_with_options(
                right.iter().map(|x| x * right_scale).collect(),
                policy,
            )
            .unwrap();
            let result = approximate_gcd(&a, &b).unwrap();
            assert_eq!(result.report().effective_degree(), degree, "{}", fields[0]);
            assert_eq!(
                result.report().sylvester_rank().map(|rank| rank.rank()),
                Some(rank),
                "{}",
                fields[0]
            );
            assert_eq!(result.polynomial().degree(), degree, "{}", fields[0]);
            assert_eq!(result.polynomial().leading_coefficient(), 1.);
            // The independently computed Jacobian condition scales binary64
            // roundoff; no binary32-sized tolerance is reused here.
            let scale = expected.iter().map(|x| x.abs()).fold(1., f64::max);
            let tolerance = 128. * f64::EPSILON * condition * scale;
            for (&actual, &oracle) in result
                .polynomial()
                .basis()
                .coefficients()
                .iter()
                .zip(&expected)
            {
                assert!(
                    (actual - oracle).abs() <= tolerance,
                    "{}: {actual} vs {oracle}, bound {tolerance}",
                    fields[0]
                );
            }
            assert!(
                result.report().backward_error() <= 16. * fit_error + tolerance,
                "{}: backward {} vs fit {}",
                fields[0],
                result.report().backward_error(),
                fit_error
            );
        }
    }
}

#[test]
fn gcd_errors_preserve_operands_and_zero_operand_is_canonical() {
    let policy = PolynomialApproximationPolicy::<f64>::default();
    let zero = ApproxPolynomial::monomial_with_options(vec![0.], policy).unwrap();
    let original = ApproxPolynomial::monomial_with_options(vec![2., 4., 2.], policy).unwrap();
    let both_zero = approximate_gcd(&zero, &zero).unwrap();
    assert!(both_zero.polynomial().is_zero());
    assert_eq!(both_zero.report().sylvester_rank(), None);
    assert_eq!(both_zero.report().reciprocal_condition(), None);
    assert_eq!(both_zero.report().operand_errors(), [0., 0.]);
    for (a, b) in [(&zero, &original), (&original, &zero)] {
        let result = approximate_gcd(a, b).unwrap();
        assert_eq!(result.polynomial().basis().coefficients(), &[1., 2., 1.]);
        assert_eq!(result.report().effective_degree(), 2);
        assert_eq!(result.report().backward_error(), 0.);
    }
    let incompatible = ApproxPolynomial::monomial_with_options(
        vec![2., 4., 2.],
        PolynomialApproximationPolicy::new(Tolerance::new(0., 1e-6).unwrap(), RankPolicy::Exact),
    )
    .unwrap();
    let snapshot = original.clone();
    assert_eq!(
        approximate_gcd(&original, &incompatible)
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::InvalidInput
    );
    assert_eq!(original, snapshot);
    assert!(zero.is_zero());
}

#[test]
fn operand_errors_are_independently_normalized_and_scale_invariant() {
    let delta = 1e-7;
    let policy = PolynomialApproximationPolicy::new(
        Tolerance::new(0., 1e-6).unwrap(),
        RankPolicy::Threshold(RankThreshold::new(0., 1e-6).unwrap()),
    );
    let mut baseline: Option<f64> = None;
    for (left_scale, right_scale) in [
        (1., 1.),
        (2.0_f64.powi(400), 2.0_f64.powi(-400)),
        (2.0_f64.powi(-400), 2.0_f64.powi(400)),
    ] {
        let a = ApproxPolynomial::monomial_with_options(
            [2., -3., 1.].map(|x| x * left_scale).to_vec(),
            policy,
        )
        .unwrap();
        let b = ApproxPolynomial::monomial_with_options(
            [3. * (1. + delta), -4. - delta, 1.]
                .map(|x| x * right_scale)
                .to_vec(),
            policy,
        )
        .unwrap();
        let result = approximate_gcd(&a, &b).unwrap();
        assert_eq!(result.report().effective_degree(), 1);
        let norm = |p: &scioxide_poly::ApproxPolynomial<f64, scioxide_poly::MonomialBasis<f64>>| {
            p.basis()
                .coefficients()
                .iter()
                .fold(0.0_f64, |a, &b| a.max(b.abs()))
        };
        let expected = [
            norm(result.remainders()[0]) / norm(&a),
            norm(result.remainders()[1]) / norm(&b),
        ];
        for (actual, expected) in result.report().operand_errors().into_iter().zip(expected) {
            assert!((actual - expected).abs() <= 8. * f64::EPSILON * expected);
        }
        assert!(
            (result.report().backward_error() - expected[0].max(expected[1])).abs()
                <= f64::EPSILON * 8. * expected[0].max(expected[1])
        );
        if let Some(error) = baseline {
            assert!((result.report().backward_error() - error).abs() <= 128. * f64::EPSILON);
        } else {
            baseline = Some(result.report().backward_error());
        }
        let rational = scioxide_poly::ApproxRationalFunction::new(a, b).unwrap();
        let simplified = rational.simplify().unwrap();
        assert_eq!(
            simplified.gcd_report().operand_errors(),
            [simplified.numerator_error(), simplified.denominator_error()]
        );
    }
}
