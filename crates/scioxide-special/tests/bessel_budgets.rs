#![allow(clippy::unwrap_used)]
#[path = "support/publication.rs"]
mod publication;
use scioxide_core::Budget;
use scioxide_special::*;
#[test]
fn shared_budget_covers_nested_and_scalar_calls() {
    for limit in [1, 2, 8, 16] {
        let options =
            BesselOptions::default().with_iteration_budget(Budget::try_from(limit).unwrap());
        for r in [
            bessel_j_with_options(2.5_f64, 2.0, options),
            bessel_y_with_options(2.5, 2.0, options),
            bessel_i_with_options(2.5, 2.0, options),
            bessel_k_with_options(2.5, 2.0, options),
        ] {
            assert!(publication::budget_counts(r).iterations() <= limit);
        }
        assert!(
            publication::budget_counts(hankel_1_with_options(2.5_f64, 2.0, options)).iterations()
                <= limit
        );
    }
}
#[test]
fn validation_and_accuracy_requests() {
    let options = BesselOptions::default().with_iteration_budget(Budget::ONE);
    {
        let outcome = &(bessel_j_with_options(f64::NAN, 2.0, options));
        let err = outcome.as_ref().expect_err("expected error");
        assert!(matches!(
            err.view(),
            scioxide_core::SciOxideErrorView::NonFinite { .. }
        ));
    }
    let r = bessel_j_with_options(0.0_f64, 2.0, BesselOptions::default()).unwrap();
    assert!(r.counts().iterations() > 0);
    assert!(r.termination().is_success());
}

#[test]
fn derivatives_and_normalized_lambda() {
    use scioxide_core::Complex;
    let z = Complex::new(2.0_f64, 1.0);
    let order = Complex::new(0.5, 0.0);
    let report = bessel_j_with_options(order, z, BesselOptions::default()).unwrap();
    let value = *report.value().unwrap().value();
    let first = *report.derivative().unwrap().value();
    let second = -first / z - (Complex::new(1.0, 0.0) - Complex::new(0.25, 0.0) / (z * z)) * value;
    assert!((bessel_j_derivative(order, z, 2).unwrap() - second).norm() < 1e-9);
    assert!((bessel_j_derivative(0.0_f64, 0.0, 4).unwrap() - 0.375).abs() < 1e-14);
    assert!((bessel_i_derivative(1.0_f64, 0.0, 5).unwrap() - 0.3125).abs() < 1e-14);
    assert_eq!(bessel_lambda(1000.0_f64, 0.0).unwrap(), 1.0);
    assert!((bessel_lambda(0.5_f64, 2.0).unwrap() - 2.0_f64.sin() / 2.0).abs() < 1e-12);
}

#[test]
fn high_derivative_order() {
    assert!((bessel_j_derivative(0.0_f64, 0.0, 2000).unwrap() - 0.01783901114585432).abs() < 1e-12);
}

#[test]
fn real_pairs_report_derivatives_and_preserve_scalar_range() {
    for x in [0.25_f64, 1.165_561_185_207_211_2, 2.0, 32.0, 800.0] {
        let scale = (2.0 / (core::f64::consts::PI * x)).sqrt();
        let options = BesselOptions::default();
        for (report, plain, derivative) in [
            (
                bessel_j_with_options(0.5, x, options).unwrap(),
                bessel_j(0.5, x).unwrap(),
                scale * (x.cos() - x.sin() / (2.0 * x)),
            ),
            (
                bessel_y_with_options(0.5, x, options).unwrap(),
                bessel_y(0.5, x).unwrap(),
                scale * (x.sin() + x.cos() / (2.0 * x)),
            ),
        ] {
            assert_eq!(report.try_value().unwrap(), plain);
            let estimate = report.derivative().unwrap();
            let error = (*estimate.value() - derivative).abs();
            assert!(
                error <= estimate.absolute_error() + 4.0 * f64::EPSILON * scale,
                "x={x}: {error} > {}",
                estimate.absolute_error()
            );
        }
    }
    let x = f32::from_bits(0x2edb_e6ff);
    let result = bessel_y_with_options(3.25_f32, x, BesselOptions::default()).unwrap();
    assert_eq!(result.try_value().unwrap(), bessel_y(3.25, x).unwrap());
    assert!(result.derivative().is_none());
    let result = bessel_j_with_options(1024.0_f64, 1042.8, BesselOptions::default()).unwrap();
    assert!(result.derivative().is_some());
    assert!(result.logarithmic().is_some());
    let x = 0.9_f32;
    let result = bessel_j_with_options(32.0_f32, x, BesselOptions::default()).unwrap();
    assert_eq!(result.try_value().unwrap(), 0.0);
    let derivative = result.derivative().unwrap();
    let expected = bessel_j_derivative(32.0_f64, x as f64, 1).unwrap() as f32;
    assert!(expected > 0.0);
    assert_eq!(*derivative.value(), expected);
    assert!(derivative.absolute_error() > 0.0);
    let tiny = bessel_j_with_options(200.0_f64, 0.01, BesselOptions::default()).unwrap();
    assert_eq!(*tiny.derivative().unwrap().value(), 0.0);
    assert!(tiny.derivative().unwrap().absolute_error() > 0.0);
}
