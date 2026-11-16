//! Public elliptic identities, domains, reports and independent fixtures.
#![allow(clippy::unwrap_used, clippy::panic)]
#[path = "support/publication.rs"]
mod publication;
#[path = "support/scaled.rs"]
mod scaled_reference;
use scioxide_core::{Budget, Complex, RealScalar};
use scioxide_special::*;

#[test]
fn complete_elliptic_domains_and_periods() {
    for m in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for result in [elliptic_k(m), elliptic_e(m)] {
            assert!(matches!(
                result.unwrap_err().view(),
                scioxide_core::SciOxideErrorView::NonFinite { .. }
            ));
        }
    }
    for result in [elliptic_k(1.0_f64.next_up()), elliptic_e(1.0_f64.next_up())] {
        assert!(matches!(
            result.unwrap_err().view(),
            scioxide_core::SciOxideErrorView::DomainError { .. }
        ));
    }
    let options = EllipticOptions::default().with_iteration_budget(Budget::ONE);
    for m in [0.25_f64, 0.9, 1.0_f64.next_down()] {
        let k = elliptic_k(m).unwrap();
        let e = elliptic_e(m).unwrap();
        for (report, expected) in [
            (
                elliptic_f_with_options(3.0 * core::f64::consts::FRAC_PI_2, m, options).unwrap(),
                3.0 * k,
            ),
            (
                elliptic_e_inc_with_options(3.0 * core::f64::consts::FRAC_PI_2, m, options)
                    .unwrap(),
                3.0 * e,
            ),
            (elliptic_pi_with_options(0.0, m, options).unwrap(), k),
            (
                elliptic_pi_with_options(m, m, options).unwrap(),
                e / (1.0 - m),
            ),
        ] {
            assert_eq!(report.counts().iterations(), 0);
            assert!(
                (report.try_value().unwrap() - expected).abs() <= 2.0 * f64::EPSILON * expected
            );
        }
    }
}

#[test]
fn carlson_degeneracies_and_limits() {
    for x in [0.0_f64, 0.1, 1.0, 5.0] {
        let rc = carlson_rc(x, 2.0).unwrap();
        assert!((rc - carlson_rf(x, 2.0, 2.0).unwrap()).abs() < 2e-14);
        assert!(
            (carlson_rd(x, 2.0, 3.0).unwrap() - carlson_rj(x, 2.0, 3.0, 3.0).unwrap()).abs()
                < 2e-14
        );
    }
    assert_eq!(carlson_rg(0.0_f64, 0.0, 0.0).unwrap(), 0.0);
    assert_eq!(carlson_rg(0.0_f64, 0.0, 4.0).unwrap(), 1.0);
    assert!((carlson_rf(1.0_f64, 2.0, 4.0).unwrap() - 0.6850858166334359).abs() < 2e-15);
    assert!((carlson_rc(0.0_f64, -1.0).unwrap()).abs() < 1e-15);
    assert!((carlson_rc(1.0_f64, -1.0).unwrap() - 0.6232252401402305).abs() < 2e-15);
}
#[test]
fn homogeneity_extends_beyond_scalar_range() {
    let expected = carlson_rd(1.0_f64, 2.0, 3.0).unwrap();
    for exponent in [-1000, 1000] {
        let scale = 2.0_f64.powi(exponent);
        let result =
            carlson_rd_with_options(scale, 2.0 * scale, 3.0 * scale, EllipticOptions::default())
                .unwrap();
        assert!(result.termination().is_success(), "{result:?}");
        let scaled = *result.scaled().unwrap().value();
        let value = scaled.try_projective().unwrap().significand()
            * 2.0_f64.powi(scaled.try_projective().unwrap().exponent() as i32 + 3 * exponent / 2);
        assert!((value - expected).abs() < 2e-14);
        // 2^-1500 rounds to zero; 2^1500 overflows.
        assert_eq!(
            result.value().map(|value| *value.value()),
            (exponent > 0).then_some(0.0)
        );
    }
}
#[test]
fn complex_symmetry_and_domains() {
    let x = Complex::new(0.5_f64, 0.25);
    let y = Complex::new(2.0, -0.5);
    let z = Complex::new(3.0, 1.0);
    assert!((carlson_rf(x, y, z).unwrap() - carlson_rf(z, x, y).unwrap()).norm() < 2e-14);
    assert!(
        (carlson_rg(x, y, z).unwrap().conj() - carlson_rg(x.conj(), y.conj(), z.conj()).unwrap())
            .norm()
            < 2e-14
    );
    {
        let outcome = &(carlson_rf(-1.0_f64, 2.0, 3.0));
        let err = outcome.as_ref().expect_err("expected error");
        assert!(matches!(
            err.view(),
            scioxide_core::SciOxideErrorView::DomainError { .. }
        ));
    }
    {
        let outcome = &(carlson_rd(1.0_f64, 2.0, 0.0));
        let err = outcome.as_ref().expect_err("expected error");
        assert!(matches!(
            err.view(),
            scioxide_core::SciOxideErrorView::DomainError { .. }
        ));
    }
    {
        let outcome = &(carlson_rc(f64::NAN, 1.0));
        let err = outcome.as_ref().expect_err("expected error");
        assert!(matches!(
            err.view(),
            scioxide_core::SciOxideErrorView::NonFinite { .. }
        ));
    }
}
#[test]
fn options_control_nested_work_and_accuracy() {
    let options = EllipticOptions::default().with_iteration_budget(Budget::ONE);
    let result = carlson_rj_with_options(1.0_f64, 2.0, 3.0, 0.25, options).unwrap_err();
    let result_counts = publication::exhausted_counts(&result);
    assert_eq!(result_counts.iterations(), 1);
    let result = carlson_rf_with_options(1.0_f64, 2.0, 3.0, EllipticOptions::default()).unwrap();
    assert_eq!(result.termination(), SpecialFunctionTermination::Converged);
}

#[test]
fn jacobi_identities_and_inverse() {
    for m in [-4.0_f64, -0.1, 0.0, 0.25, 0.9, 1.0, 2.0, 10.0] {
        for u in [-3.0_f64, -0.7, 0.0, 0.1, 0.8, 2.0, 5.0] {
            let sn = jacobi_sn(u, m).unwrap();
            let cn = jacobi_cn(u, m).unwrap();
            let dn = jacobi_dn(u, m).unwrap();
            assert!((sn * sn + cn * cn - 1.0).abs() < 3e-14, "u={u} m={m}");
            assert!((dn * dn + m * sn * sn - 1.0).abs() < 3e-14, "u={u} m={m}");
            let am = jacobi_amplitude(u, m).unwrap();
            assert!((am.sin() - sn).abs() < 3e-14);
            assert!((am.cos() - cn).abs() < 3e-14);
            if m < 1.0 {
                assert!((elliptic_f(am, m).unwrap() - u).abs() < 1e-12);
            }
        }
    }
}
#[test]
fn complex_jacobi_identities() {
    for m in [-3.0_f64, 0.0, 0.25, 0.9, 1.0, 2.0] {
        for u in [Complex::new(0.3_f64, 0.4), Complex::new(-1.0, -0.7)] {
            let sn = jacobi_sn(u, m).unwrap();
            let cn = jacobi_cn(u, m).unwrap();
            let dn = jacobi_dn(u, m).unwrap();
            assert!(
                (sn * sn + cn * cn - Complex::new(1.0, 0.0)).norm() < 1e-12,
                "u={u:?} m={m}"
            );
            assert!(
                (dn * dn + sn * sn * Complex::new(m, 0.0) - Complex::new(1.0, 0.0)).norm() < 1e-12
            );
            let am = jacobi_amplitude(u, m).unwrap();
            assert!((am.sin() - sn).norm() < 1e-12);
            assert!((am.cos() - cn).norm() < 1e-12);
        }
    }
}
#[test]
fn jacobi_f32_and_periods() {
    let k = elliptic_k(0.5_f64).unwrap();
    for u in [0.1_f64, 0.8, 1.5] {
        assert!((jacobi_sn(u, 0.5).unwrap() + jacobi_sn(u + 2.0 * k, 0.5).unwrap()).abs() < 1e-14);
        assert!(
            (jacobi_amplitude(u + 2.0 * k, 0.5).unwrap()
                - jacobi_amplitude(u, 0.5).unwrap()
                - core::f64::consts::PI)
                .abs()
                < 1e-14
        );
        let v = jacobi_sn(u as f32, 0.5_f32).unwrap();
        assert!((f64::from(v) - jacobi_sn(u as f32 as f64, 0.5).unwrap()).abs() < 1e-7);
    }
    assert!(jacobi_sn(-0.0_f64, 0.5).unwrap().is_sign_negative());
}

fn fixture_result<T: scioxide_core::NumericScalar>(
    family: &str,
    args: [T; 4],
) -> SpecialFunctionResult<T> {
    let options = EllipticOptions::new(Budget::try_from(512).unwrap());
    match family {
        "rf" => carlson_rf_with_options(args[0], args[1], args[2], options),
        "rd" => carlson_rd_with_options(args[0], args[1], args[2], options),
        "rg" => carlson_rg_with_options(args[0], args[1], args[2], options),
        "rc" => carlson_rc_with_options(args[0], args[1], options),
        "rj" => carlson_rj_with_options(args[0], args[1], args[2], args[3], options),
        "sn" => jacobi_sn_with_options(args[0], T::from_real(args[1].real_component()), options),
        "cn" => jacobi_cn_with_options(args[0], T::from_real(args[1].real_component()), options),
        "dn" => jacobi_dn_with_options(args[0], T::from_real(args[1].real_component()), options),
        "am" => {
            jacobi_amplitude_with_options(args[0], T::from_real(args[1].real_component()), options)
        }
        _ => unreachable!("unknown fixture family"),
    }
    .unwrap_or_else(|e| panic!("{family} {args:?}: {e}"))
}
fn check_fixture<T: scioxide_core::NumericScalar>(row: &[&str]) {
    let args = core::array::from_fn(|i| {
        let (re, im) = row[i + 3].split_once(',').unwrap();
        T::from_components(
            T::Real::try_from_f64_lossy(re.parse().unwrap()).unwrap(),
            T::Real::try_from_f64_lossy(im.parse().unwrap()).unwrap(),
        )
    });
    let result = fixture_result(row[2], args);
    assert!(
        result.termination().is_success(),
        "{} {:?}",
        row.join(" "),
        result
    );
    let scaled = *result.scaled().unwrap().value();
    let exponent: i64 = row[7].parse().unwrap();
    let actual = scaled_reference::at_exponent(&scaled, exponent);
    let (re, im) = (actual.re, actual.im);
    let tolerance = if row[0] == "32" { 3e-6 } else { 5e-12 };
    let bounds: Vec<f64> = row[8..].iter().map(|s| s.parse().unwrap()).collect();
    for (actual, pair) in [re, im].into_iter().zip(bounds.chunks(2)) {
        let error = (pair[0] - actual).max(actual - pair[1]).max(0.0);
        assert!(
            error <= tolerance,
            "{} error={error:e} actual={actual:e}",
            row.join(" ")
        );
        let estimated = result
            .scaled()
            .unwrap()
            .absolute_error()
            .try_to_f64_exact()
            .unwrap()
            .scalbn(
                (i128::from(scaled.scale_exponent()) - i128::from(exponent))
                    .clamp(i128::from(i32::MIN), i128::from(i32::MAX)) as i32,
            );
        assert!(
            error <= estimated + 8.0 * f64::EPSILON * actual.abs(),
            "{} error={error:e} estimated={estimated:e}",
            row.join(" ")
        );
    }
}
#[test]
fn arb_and_theta_reference_enclosures() {
    for line in include_str!("data/elliptic.tsv")
        .lines()
        .filter(|s| !s.starts_with('#'))
    {
        let row: Vec<_> = line.split_whitespace().collect();
        match (row[0], row[1]) {
            ("32", "real") => check_fixture::<f32>(&row),
            ("64", "real") => check_fixture::<f64>(&row),
            ("32", "complex") => check_fixture::<Complex<f32>>(&row),
            ("64", "complex") => check_fixture::<Complex<f64>>(&row),
            _ => unreachable!("unknown fixture scalar"),
        }
    }
}

#[test]
fn duplication_and_total_budget_contracts() {
    for [x, y, z, p] in [[0.0_f64, 0.5, 1.0, 0.3], [1.0, 2.0, 3.0, 0.25]] {
        let lambda = x.sqrt() * y.sqrt() + y.sqrt() * z.sqrt() + z.sqrt() * x.sqrt();
        let [xx, yy, zz, pp] = [x, y, z, p].map(|a| (a + lambda) * 0.25);
        assert!((carlson_rf(x, y, z).unwrap() - carlson_rf(xx, yy, zz).unwrap()).abs() < 1e-14);
        let rd = 0.25 * carlson_rd(xx, yy, zz).unwrap() + 3.0 / (z.sqrt() * (z + lambda));
        assert!((carlson_rd(x, y, z).unwrap() - rd).abs() < 2e-14);
        let alpha = p * (x.sqrt() + y.sqrt() + z.sqrt()) + (x * y * z).sqrt();
        let beta = p * (p + lambda).powi(2);
        let rj = 0.25 * carlson_rj(xx, yy, zz, pp).unwrap()
            + 3.0 * carlson_rc(alpha * alpha, beta).unwrap();
        assert!((carlson_rj(x, y, z, p).unwrap() - rj).abs() < 2e-14);
    }
    let options = EllipticOptions::default().with_iteration_budget(Budget::ONE);
    for result in [
        elliptic_f_with_options(4.0_f64, 0.5, options),
        elliptic_e_inc_with_options(4.0, 0.5, options),
    ] {
        assert!(publication::exhausted_counts(&result.unwrap_err()).iterations() <= 1);
    }
    assert_eq!(
        elliptic_k_with_options(0.5_f64, EllipticOptions::default())
            .unwrap()
            .termination(),
        SpecialFunctionTermination::Converged
    );
    // A huge argument loses the phase; its derived error reports that loss.
    let large = jacobi_sn_with_options(1e20_f64, 0.5, EllipticOptions::default()).unwrap();
    assert_eq!(large.termination(), SpecialFunctionTermination::Converged);
    assert!(large.value().unwrap().absolute_error() >= 1.0);
}

#[test]
fn real_carlson_dispatch_preserves_homogeneity_and_projection() {
    let options = EllipticOptions::default();
    let reference = [
        carlson_rf(1.0_f64, 2.0, 4.0).unwrap(),
        carlson_rd(1.0_f64, 2.0, 4.0).unwrap(),
        carlson_rj(1.0_f64, 2.0, 4.0, 0.5).unwrap(),
        carlson_rc(1.0_f64, -2.0).unwrap(),
        carlson_rg(1.0_f64, 2.0, 4.0).unwrap(),
        carlson_rj(1.0_f64, 2.0, 4.0, -0.5).unwrap(),
    ];
    // Both sides of the cubic-intermediate range guard, and scaled-only RD/RJ.
    for exponent in [-1000, -302, -300, -298, 0, 298, 300, 302, 1000] {
        for scale in [
            2.0_f64.powi(exponent).next_down(),
            2.0_f64.powi(exponent),
            2.0_f64.powi(exponent).next_up(),
        ] {
            let reports = [
                carlson_rf_with_options(scale, 2.0 * scale, 4.0 * scale, options).unwrap(),
                carlson_rd_with_options(scale, 2.0 * scale, 4.0 * scale, options).unwrap(),
                carlson_rj_with_options(scale, 2.0 * scale, 4.0 * scale, 0.5 * scale, options)
                    .unwrap(),
                carlson_rc_with_options(scale, -2.0 * scale, options).unwrap(),
                carlson_rg_with_options(scale, 2.0 * scale, 4.0 * scale, options).unwrap(),
                carlson_rj_with_options(scale, 2.0 * scale, 4.0 * scale, -0.5 * scale, options)
                    .unwrap(),
            ];
            for ((report, expected), degree) in reports
                .into_iter()
                .zip(reference)
                .zip([-1, -3, -3, -1, 1, -3])
            {
                assert!(report.termination().is_success());
                let value = scaled_reference::at_exponent(
                    report.scaled().unwrap().value(),
                    i64::from(degree * exponent / 2),
                )
                .re;
                let multiplier = (scale.scalbn(-exponent)).sqrt().powi(degree);
                assert!(
                    (value - expected * multiplier).abs() < 8e-14 * expected.abs(),
                    "exponent={exponent}, degree={degree}"
                );
            }
        }
    }
}

#[test]
fn real_jacobi_limits_and_transformation_boundaries() {
    for m in [
        -10.0_f64,
        -1.0,
        -f64::MIN_POSITIVE,
        0.0,
        f64::MIN_POSITIVE,
        1.0_f64.next_down(),
        1.0,
        1.0_f64.next_up(),
        2.0,
    ] {
        for u in [-2.0_f64, -0.0, f64::from_bits(1), 0.125, 2.0] {
            let sn = jacobi_sn(u, m).unwrap();
            let cn = jacobi_cn(u, m).unwrap();
            let dn = jacobi_dn(u, m).unwrap();
            assert!((sn * sn + cn * cn - 1.0).abs() < 4e-14);
            assert!((dn * dn + m * sn * sn - 1.0).abs() < 4e-14);
            for (actual, detailed) in [
                (
                    sn,
                    jacobi_sn_with_options(u, m, EllipticOptions::default()).unwrap(),
                ),
                (
                    cn,
                    jacobi_cn_with_options(u, m, EllipticOptions::default()).unwrap(),
                ),
                (
                    dn,
                    jacobi_dn_with_options(u, m, EllipticOptions::default()).unwrap(),
                ),
            ] {
                assert_eq!(
                    actual.to_bits(),
                    detailed.value().unwrap().value().to_bits()
                );
            }
        }
    }
    for component in [jacobi_cn_with_options, jacobi_dn_with_options] {
        // cn(1000|1) = dn(1000|1) = sech(1000) rounds to zero.
        let report = component(1000.0_f64, 1.0, EllipticOptions::default()).unwrap();
        assert_eq!(report.try_value().unwrap().to_bits(), 0);
        assert!(report.scaled().is_some());
        assert!(report.termination().is_success());
    }
}

#[test]
fn subnormal_projection_rounding_is_reported_separately() {
    let scale = 2.0_f32.powi(96);
    let result =
        carlson_rd_with_options(scale, 2.0 * scale, 3.0 * scale, EllipticOptions::default())
            .unwrap();
    assert_eq!(result.termination(), SpecialFunctionTermination::Converged);
    let actual = f64::from(*result.value().unwrap().value());
    let reference = carlson_rd(1.0_f64, 2.0, 3.0).unwrap() * 2.0_f64.powi(-144);
    let error = (actual - reference).abs();
    assert!(error > 1e-5 * reference);
    assert!(f64::from(result.value().unwrap().absolute_error()) >= error);
}

#[test]
fn subnormal_jacobi_rounding_is_not_an_exact_identity() {
    let options = EllipticOptions::default();
    let tiny = f64::from_bits(1);
    let sn = jacobi_sn_with_options(tiny, 0.5, options).unwrap();
    assert_eq!(sn.termination(), SpecialFunctionTermination::Converged);
    assert_eq!(*sn.value().unwrap().value(), tiny);
    assert!(sn.value().unwrap().absolute_error() >= tiny);
    assert_eq!(
        jacobi_sn_with_options(-0.0_f64, 0.5, options)
            .unwrap()
            .termination(),
        SpecialFunctionTermination::Exact
    );
    assert_eq!(
        jacobi_dn_with_options(0.25_f64, 0.0, options)
            .unwrap()
            .termination(),
        SpecialFunctionTermination::Exact
    );
}
