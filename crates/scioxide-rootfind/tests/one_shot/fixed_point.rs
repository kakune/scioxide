//! `fixed_point`.
use scioxide_core::{
    Complex, ConvergenceReason, NonFiniteOrigin, NumericScalar, RealScalar, SciOxideError,
    SciOxideErrorView, SolveCounts,
};
use scioxide_rootfind::{
    FixedPointInput, FixedPointMethod, FixedPointOptions, FixedPointResult, FixedPointSolver,
    ResidualFn, RootTermination, fixed_point,
};

fn non_finite(error: &SciOxideError) -> Option<NonFiniteOrigin> {
    match error.view() {
        SciOxideErrorView::NonFinite { origin, .. } => Some(origin),
        _ => None,
    }
}

fn convergence_failure(error: &SciOxideError) -> Option<(ConvergenceReason, SolveCounts)> {
    match error.view() {
        SciOxideErrorView::ConvergenceFailure {
            reason,
            diagnostics,
            ..
        } => Some((reason, diagnostics.counts())),
        _ => None,
    }
}

type Map<T> = fn(&[T], &mut [T]) -> Result<(), SciOxideError>;

/// `G(x) = x/2 + 1`, componentwise, with the fixed point `x = 2`.
fn halving<T: NumericScalar>(x: &[T], out: &mut [T]) -> Result<(), SciOxideError> {
    let half = T::from_real(T::Real::HALF);
    for (slot, &value) in out.iter_mut().zip(x) {
        *slot = half * value + T::ONE;
    }
    Ok(())
}

/// `G(x, y) = (cos y, sin x / 2)`, a contraction of the plane.
fn trigonometric(x: &[f64], out: &mut [f64]) -> Result<(), SciOxideError> {
    out[0] = x[1].cos();
    out[1] = 0.5 * x[0].sin();
    Ok(())
}

/// `G(z) = (1/4 + i/8) z + 2 − i`, whose fixed point is
/// `z* = (2 − i)/(3/4 − i/8)`.
fn complex_affine(z: &[Complex<f64>], out: &mut [Complex<f64>]) -> Result<(), SciOxideError> {
    out[0] = Complex::new(0.25, 0.125) * z[0] + Complex::new(2.0, -1.0);
    Ok(())
}

fn configured<T: NumericScalar>(g: Map<T>, initial: &[T]) -> FixedPointResult<T> {
    FixedPointSolver::<T>::new(
        FixedPointMethod::Anderson,
        FixedPointInput::Map,
        FixedPointOptions::default(),
    )
    .unwrap()
    .solve(&ResidualFn::new(initial.len(), g), initial)
    .unwrap()
}

fn matches_anderson<T: NumericScalar>(g: Map<T>, initial: &[T]) {
    let configured = configured(g, initial);
    assert!(configured.is_success(), "{configured:?}");
    let one_shot = fixed_point(g, initial).unwrap();
    assert_eq!(one_shot, configured.point());
}

#[test]
fn fixed_point_equals_anderson_with_default_options() {
    matches_anderson::<f64>(halving, &[0.0]);
    matches_anderson::<f32>(halving, &[0.0, -3.0]);
    matches_anderson::<f64>(trigonometric, &[0.0, 0.0]);
    matches_anderson::<Complex<f64>>(complex_affine, &[Complex::new(0.0, 0.0)]);
    matches_anderson::<Complex<f64>>(halving, &[Complex::new(1.0, -1.0); 3]);
}

#[test]
fn fixed_point_reaches_reference_values() {
    let x = fixed_point(halving::<f64>, &[0.0]).unwrap();
    assert!((x[0] - 2.0).abs() < 1e-7, "{x:?}");
    let x = fixed_point(halving::<f32>, &[0.0]).unwrap();
    assert!((x[0] - 2.0).abs() < 1e-3, "{x:?}");

    let x = fixed_point(trigonometric, &[0.0, 0.0]).unwrap();
    assert!((x[0] - x[1].cos()).abs() < 1e-7 && (x[1] - 0.5 * x[0].sin()).abs() < 1e-7);

    let z = fixed_point(complex_affine, &[Complex::new(0.0, 0.0)]).unwrap();
    let expected = Complex::new(2.0, -1.0) / Complex::new(0.75, -0.125);
    assert!((z[0] - expected).norm() < 1e-7, "{z:?}");
}

#[test]
fn fixed_point_rejects_non_finite_starting_points() {
    let error = fixed_point(halving::<f64>, &[f64::NAN]).unwrap_err();
    assert_eq!(non_finite(&error), Some(NonFiniteOrigin::Input));
    let error = fixed_point(halving::<f32>, &[0.0, f32::INFINITY]).unwrap_err();
    assert_eq!(non_finite(&error), Some(NonFiniteOrigin::Input));
    let error = fixed_point(complex_affine, &[Complex::new(0.0, f64::NAN)]).unwrap_err();
    assert_eq!(non_finite(&error), Some(NonFiniteOrigin::Input));
}

#[test]
fn fixed_point_returns_callback_errors_unchanged() {
    let failure = SciOxideError::external("callback failed");
    let error = fixed_point(
        |_: &[f64], _: &mut [f64]| Err(SciOxideError::external("callback failed")),
        &[0.0],
    );
    assert_eq!(error.unwrap_err(), failure);

    let error = fixed_point(
        |_: &[f64], out: &mut [f64]| {
            out[0] = f64::INFINITY;
            Ok(())
        },
        &[0.0],
    );
    assert_eq!(
        non_finite(&error.unwrap_err()),
        Some(NonFiniteOrigin::Callback)
    );
}

/// `G(x) = x + 1` has no fixed point: the residual is always 1 and every step
/// has length 1, so the configured form stops at its 200-iteration limit.
#[test]
fn fixed_point_reports_an_unsuccessful_termination_as_an_error() {
    let shift: Map<f64> = |x, out| {
        out[0] = x[0] + 1.0;
        Ok(())
    };
    let result = configured(shift, &[0.0]);
    assert_eq!(result.termination(), RootTermination::IterationLimit);
    assert_eq!(result.iterations(), 200);
    let error = fixed_point(shift, &[0.0]).unwrap_err();
    assert_eq!(
        convergence_failure(&error),
        Some((ConvergenceReason::IterationLimit, result.accounting()))
    );
}

#[test]
fn fixed_point_accepts_a_fixed_start_and_an_empty_start() {
    assert_eq!(
        fixed_point(halving::<f64>, &[2.0, 2.0]).unwrap(),
        [2.0, 2.0]
    );
    let start = [Complex::new(2.0, 0.0)];
    assert_eq!(fixed_point(halving::<Complex<f64>>, &start).unwrap(), start);
    assert!(fixed_point(halving::<f64>, &[]).unwrap().is_empty());
    assert!(fixed_point(halving::<f32>, &[]).unwrap().is_empty());
}
