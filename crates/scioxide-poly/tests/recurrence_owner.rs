//! Shared scalar Legendre and symmetric orthogonal recurrence contracts.
#![allow(clippy::unwrap_used)]

use scioxide_core::{NonFiniteOrigin, RealScalar, Scaled, SciOxideErrorView};
use scioxide_poly::{ApproxPolynomial, BasisNormalization, OrthogonalBasisSpec, OrthogonalFamily};

#[test]
fn scalar_legendre_matches_independent_references_through_degree_ten_thousand() {
    fn check<R: RealScalar>(scalar: &str, decode: impl Fn(&str) -> R) {
        // The special-function crate owns these references; read them in place.
        let conditions =
            include_str!("../../scioxide-special/tests/data/legendre_ownership_conditions.tsv");
        for row in include_str!("../../scioxide-special/tests/data/legendre_ownership.tsv")
            .lines()
            .filter(|row| !row.starts_with('#'))
        {
            let fields: Vec<_> = row.split_whitespace().collect();
            if fields[1] != scalar {
                continue;
            }
            let encoded = fields[3];
            let (degree, point) = encoded.split_once(',').unwrap();
            let degree = degree.strip_prefix("u:").unwrap().parse().unwrap();
            let x = decode(point.strip_prefix("f:").unwrap());
            let expected = decode(fields[4]);
            let spacing = expected.abs().next_after(R::INFINITY) - expected.abs();
            let conditioned = conditions
                .lines()
                .filter(|row| !row.starts_with('#'))
                .find_map(|row| {
                    let fields: Vec<_> = row.split_whitespace().collect();
                    (fields[0] == scalar && fields[2] == encoded)
                        .then(|| fields[3].parse::<f64>().unwrap())
                })
                .unwrap_or(0.0);
            let bound = (R::try_from_f64_exact(64.0).unwrap() * spacing)
                .max(R::try_from_f64_lossy(conditioned).unwrap());
            let canonical = scioxide_special::legendre_p(degree, x).unwrap();
            let family = OrthogonalFamily::Legendre.evaluate(degree, x).unwrap();
            assert!(
                (canonical - expected).abs() <= bound,
                "{row}: {canonical:?}, bound {bound:?}"
            );
            assert!(
                (family - expected).abs() <= bound,
                "{row}: {family:?}, bound {bound:?}"
            );
            assert_eq!(canonical, family);
        }
    }
    check("f64", |bits| {
        f64::from_bits(u64::from_str_radix(bits, 16).unwrap())
    });
    check("f32", |bits| {
        f32::from_bits(u32::from_str_radix(bits, 16).unwrap())
    });
}

#[test]
fn family_legendre_inherits_canonical_degree_and_work_boundaries() {
    let family = OrthogonalFamily::Legendre;
    assert_eq!(family.evaluate(usize::MAX, -1.0_f64).unwrap(), -1.0);
    assert!(matches!(
        family.evaluate(usize::MAX, 0.3_f64).unwrap_err().view(),
        SciOxideErrorView::InvalidInput { .. }
    ));
    let degree = scioxide_special::LegendreOptions::default()
        .iteration_budget()
        .get()
        + 2;
    assert!(matches!(
        family.evaluate(degree, 0.3_f64).unwrap_err().view(),
        SciOxideErrorView::ConvergenceFailure { .. }
    ));
    assert!(family.evaluate(2, f64::MAX).is_err());
    assert_eq!(family.evaluate(3, -2.0_f64).unwrap(), -17.0);
}

#[test]
fn symmetric_members_match_orthonormal_calculus_conventions_in_both_widths() {
    fn check<R: RealScalar>() {
        let r = |x| R::try_from_f64_lossy(x).unwrap();
        for family in [
            OrthogonalFamily::ChebyshevT,
            OrthogonalFamily::ChebyshevU,
            OrthogonalFamily::Legendre,
            OrthogonalFamily::Jacobi {
                alpha: r(-0.75),
                beta: r(-0.25),
            },
            OrthogonalFamily::Jacobi {
                alpha: r(0.25),
                beta: r(0.75),
            },
            OrthogonalFamily::Gegenbauer { lambda: r(-0.25) },
            OrthogonalFamily::Gegenbauer { lambda: r(1.5) },
            OrthogonalFamily::Laguerre,
            OrthogonalFamily::GeneralizedLaguerre { alpha: r(-0.75) },
            OrthogonalFamily::HermiteH,
            OrthogonalFamily::HermiteHe,
        ] {
            let shift = r(0.25);
            let scale = r(1.5);
            let spec =
                OrthogonalBasisSpec::new(family, BasisNormalization::Orthonormal, shift, scale)
                    .unwrap();
            let mass = family.log_weight_mass().unwrap().exp() * scale;
            let mut sign = R::ONE;
            for degree in 0..9 {
                let recurrence = family.jacobi_recurrence(degree + 1).unwrap();
                let mut coefficients = vec![R::ZERO; degree + 1];
                coefficients[degree] = R::ONE;
                let basis_plan = spec
                    .prepare(coefficients.len() - 1, Default::default())
                    .unwrap();
                let polynomial = ApproxPolynomial::orthogonal_with_plan(
                    coefficients,
                    basis_plan,
                    Default::default(),
                )
                .unwrap();
                for x in [r(-0.75), r(0.0), r(0.3), r(1.25)] {
                    let actual = recurrence.last_member_scaled(x).unwrap().round().unwrap();
                    let expected =
                        polynomial.evaluate(shift + scale * x).unwrap() * mass.sqrt() * sign;
                    assert!(
                        (actual - expected).abs()
                            <= r(512.0) * R::EPSILON * expected.abs().max(R::ONE),
                        "{family:?}, degree {degree}, x {x:?}: {actual:?} != {expected:?}"
                    );
                }
                if family.recurrence_coefficients(degree).unwrap().0 < R::ZERO {
                    sign = -sign;
                }
            }
        }
    }
    check::<f32>();
    check::<f64>();
}

#[test]
fn scaled_members_retain_range_and_execute() {
    let recurrence = OrthogonalFamily::Legendre.jacobi_recurrence(3).unwrap();
    let x = 1e200_f64;
    let actual = recurrence.last_member_scaled(x).unwrap();
    assert!(actual.round().is_err());
    let expected = Scaled::from_value(x)
        .unwrap()
        .square()
        .unwrap()
        .mul(Scaled::from_value(3.0 * 5.0_f64.sqrt() / 2.0).unwrap())
        .unwrap();
    assert!((actual.div(expected).unwrap().round().unwrap() - 1.0).abs() < 16.0 * f64::EPSILON);
    let constant = OrthogonalFamily::HermiteH.jacobi_recurrence(1).unwrap();
    assert_eq!(
        constant
            .last_member_scaled(f64::MAX)
            .unwrap()
            .round()
            .unwrap(),
        1.0
    );
    for x in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(matches!(
            constant.last_member_scaled(x).unwrap_err().view(),
            SciOxideErrorView::NonFinite {
                origin: NonFiniteOrigin::Input,
                ..
            }
        ));
    }
}
