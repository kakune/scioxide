#![allow(clippy::unwrap_used)]
#[path = "support/errors.rs"]
mod errors;
use errors::error_kind;
use scioxide_core::{Complex, NumericScalar, RealScalar, SciOxideErrorKind};
use scioxide_signal::*;
fn s<S: RealScalar>(v: f64) -> S {
    S::try_from_f64_lossy(v).unwrap()
}
fn polynomials<T: NumericScalar>() {
    let tol = if core::mem::size_of::<T::Real>() == 4 {
        3e-4
    } else {
        2e-11
    };
    for (window, position) in [(9, None), (8, Some(3.5)), (8, Some(1.))] {
        for derivative in 0..=3 {
            let mut options = SavitzkyGolayOptions::new()
                .with_derivative(derivative)
                .with_spacing(s::<T::Real>(0.25));
            if let Some(position) = position {
                options = options.with_position(s(position));
            }
            let p = SavitzkyGolayPlan::new(window, 3, options).unwrap();
            let x: Vec<T> = (0..40)
                .map(|i| {
                    let t = i as f64 * 0.25;
                    T::from_components(s(t * t * t / 100.), s(t * t / 100.))
                })
                .collect();
            let y = p.filter(&x, SavitzkyGolayBoundary::Polynomial).unwrap();
            for (i, v) in y.iter().enumerate() {
                let t = (i as f64 + position.map_or(0., |p| p - p.floor())) * 0.25;
                let real = match derivative {
                    0 => t * t * t / 100.,
                    1 => 3. * t * t / 100.,
                    2 => 6. * t / 100.,
                    _ => 0.06,
                };
                let imaginary = match derivative {
                    0 => t * t / 100.,
                    1 => 2. * t / 100.,
                    2 => 0.02,
                    _ => 0.,
                };
                let expected = T::from_components(s(real), s(imaginary));
                assert!(
                    (v.real_component() - expected.real_component())
                        .abs()
                        .into_f64()
                        < tol,
                    "{window} {derivative} {i}: {:?} {:?}",
                    v,
                    expected
                );
                assert!(
                    (v.imaginary_component() - expected.imaginary_component())
                        .abs()
                        .into_f64()
                        < tol
                );
            }
        }
    }
}
#[test]
fn real64() {
    polynomials::<f64>();
}
#[test]
fn real32() {
    polynomials::<f32>();
}
#[test]
fn complex64() {
    polynomials::<Complex<f64>>();
}
#[test]
fn complex32() {
    polynomials::<Complex<f32>>();
}
// Independent projection onto cubics by twice-applied modified Gram–Schmidt.
fn projection(window: &[f64]) -> Vec<f64> {
    let center = (window.len() - 1) as f64 / 2.0;
    let mut basis: Vec<Vec<f64>> = Vec::new();
    for k in 0..4 {
        let mut v: Vec<f64> = (0..window.len())
            .map(|j| ((j as f64 - center) / center).powi(k))
            .collect();
        for _ in 0..2 {
            for q in &basis {
                let dot: f64 = q.iter().zip(&v).map(|(a, b)| a * b).sum();
                v.iter_mut().zip(q).for_each(|(v, q)| *v -= dot * q);
            }
        }
        let norm = v.iter().map(|v| v * v).sum::<f64>().sqrt();
        basis.push(v.into_iter().map(|v| v / norm).collect());
    }
    let mut fitted = vec![0.0; window.len()];
    for q in &basis {
        let dot: f64 = q.iter().zip(window).map(|(a, b)| a * b).sum();
        fitted.iter_mut().zip(q).for_each(|(f, q)| *f += dot * q);
    }
    fitted
}
#[test]
fn wide_polynomial_boundary_matches_independent_least_squares() {
    let window = 4001;
    let plan = SavitzkyGolayPlan::new(window, 3, SavitzkyGolayOptions::<f64>::new()).unwrap();
    let len = window + 500;
    let x: Vec<f64> = (0..len)
        .map(|j| {
            let t = j as f64;
            (0.003 * t).sin() + 0.1 * (0.05 * t).cos() + ((j * 7919) % 101) as f64 / 1000.0
        })
        .collect();
    let y = plan.filter(&x, SavitzkyGolayBoundary::Polynomial).unwrap();
    let first = projection(&x[..window]);
    let last = projection(&x[len - window..]);
    let half = window / 2;
    for i in [0, 1, 17, half - 1] {
        assert!((y[i] - first[i]).abs() < 1e-12, "start {i}");
        let k = len - 1 - i;
        assert!((y[k] - last[window - 1 - i]).abs() < 1e-12, "end {k}");
    }
    // Interior outputs project their own centered window.
    let middle = projection(&x[250..250 + window]);
    assert!((y[250 + half] - middle[half]).abs() < 1e-12);
}
#[test]
fn exact_stencil_and_validation() {
    let options = SavitzkyGolayOptions::new().with_spacing(1.);
    assert_eq!(options.position(), None);
    assert_eq!(options.derivative(), 0);
    let p = SavitzkyGolayPlan::new(5, 2, options).unwrap();
    for (&a, b) in p.coefficients().iter().zip([-3., 12., 17., 12., -3.]) {
        assert!((a - b / 35.).abs() < 1e-14);
    }
    assert_eq!(
        error_kind(SavitzkyGolayPlan::new(
            4,
            2,
            SavitzkyGolayOptions::<f64>::new()
        )),
        SciOxideErrorKind::InvalidInput
    );
    assert_eq!(
        error_kind(SavitzkyGolayPlan::new(5, 2, options.with_derivative(3))),
        SciOxideErrorKind::InvalidInput
    );
    assert_eq!(
        error_kind(SavitzkyGolayPlan::new(5, 2, options.with_spacing(0.))),
        SciOxideErrorKind::InvalidInput
    );
    assert_eq!(
        error_kind(p.filter(&[1., 2.], SavitzkyGolayBoundary::Polynomial)),
        SciOxideErrorKind::InvalidInput
    );
    assert!(
        p.filter::<f64>(&[], SavitzkyGolayBoundary::Polynomial)
            .unwrap()
            .is_empty()
    );
}
#[test]
fn default_options_smooth_at_the_midpoint_and_the_plan_keeps_its_lengths() {
    const NEW: SavitzkyGolayOptions<f64> = SavitzkyGolayOptions::new();
    let default = SavitzkyGolayOptions::<f64>::default();
    assert_eq!((default.derivative(), NEW.derivative()), (0, 0));
    assert_eq!((default.spacing(), NEW.spacing()), (1., 1.));
    assert_eq!((default.position(), NEW.position()), (None, None));
    assert_eq!(default.execution_policy(), NEW.execution_policy());
    let plan = SavitzkyGolayPlan::new(7, 3, default).unwrap();
    assert_eq!((plan.window_length(), plan.polynomial_degree()), (7, 3));
    let x: Vec<f64> = (0..12).map(|i| (i * i) as f64).collect();
    let y = plan.filter(&x, SavitzkyGolayBoundary::Polynomial).unwrap();
    assert!(
        y.iter()
            .zip(&x)
            .all(|(a, b)| (a - b).abs() <= 1e-12 * (1. + b.abs()))
    );
}
