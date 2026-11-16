//! SciPy `savgol_coeffs` and `savgol_filter` references.
//!
//! Fixture: `fixtures/savgol_scipy.tsv`, written by
//! `generate_savgol.py` (mode mapping in its
//! docstring). Stored values are exact rational results rounded once, checked
//! against SciPy by the generator. Errors are normwise: the largest value error divided by
//! `max(max |reference|, 1)`.
#![allow(clippy::unwrap_used)]
use scioxide_core::{RealScalar, SciOxideErrorView};
use scioxide_signal::*;

fn numbers(line: &str) -> Vec<f64> {
    line.split_whitespace()
        .map(|x| x.parse().unwrap())
        .collect()
}
fn s<S: RealScalar>(v: f64) -> S {
    S::try_from_f64_lossy(v).unwrap()
}
fn normwise<S: RealScalar>(actual: &[S], expected: &[f64]) -> f64 {
    assert_eq!(actual.len(), expected.len());
    let scale = expected.iter().fold(1.0_f64, |m, x| m.max(x.abs()));
    actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a.into_f64() - b).abs())
        .fold(0.0, f64::max)
        / scale
}
/// Window length, polynomial degree and options of a fixture header.
fn options<S: RealScalar>(header: &[&str]) -> (usize, usize, SavitzkyGolayOptions<S>) {
    let field = |i: usize| -> f64 { header[i].parse().unwrap() };
    let options = SavitzkyGolayOptions::new()
        .with_derivative(field(3) as usize)
        .with_spacing(s(field(4)));
    (field(1) as usize, field(2) as usize, options)
}

/// f64 meets 1e-12 (observed worst 1.0e-15). Inputs, spacings and positions
/// are exact in f32, so f32 errors are the fit and dot-product rounding
/// alone: observed at most 6.7 epsilon, bounded here by 64 epsilon.
fn reference<S: RealScalar>() {
    let tolerance = if S::MANTISSA_DIGITS == 24 {
        64.0 * S::EPSILON.into_f64()
    } else {
        1e-12
    };
    let lines: Vec<_> = include_str!("fixtures/savgol_scipy.tsv")
        .lines()
        .filter(|s| !s.starts_with('#'))
        .collect();
    let mut cursor = 0;
    while cursor < lines.len() {
        let header: Vec<_> = lines[cursor].split_whitespace().collect();
        let (error, consumed) = match header[0] {
            "coeffs" => {
                let (window, degree, options) = options::<S>(&header);
                let options = options.with_position(s(header[5].parse().unwrap()));
                let plan = SavitzkyGolayPlan::new(window, degree, options).unwrap();
                (
                    normwise(plan.coefficients(), &numbers(lines[cursor + 1])),
                    2,
                )
            }
            record => {
                assert_eq!(record, "filter");
                let boundary = match header[5] {
                    "interp" => SavitzkyGolayBoundary::Polynomial,
                    "constant" => SavitzkyGolayBoundary::Extend(SignalBoundary::Zero),
                    "wrap" => SavitzkyGolayBoundary::Extend(SignalBoundary::Periodic),
                    mode => {
                        assert_eq!(mode, "mirror");
                        SavitzkyGolayBoundary::Extend(SignalBoundary::Reflect)
                    }
                };
                let input: Vec<S> = numbers(lines[cursor + 1]).into_iter().map(s).collect();
                assert_eq!(input.len(), header[6].parse::<usize>().unwrap());
                let (window, degree, options) = options::<S>(&header);
                let plan = SavitzkyGolayPlan::new(window, degree, options).unwrap();
                let output = plan.filter(&input, boundary).unwrap();
                (normwise(&output, &numbers(lines[cursor + 2])), 3)
            }
        };
        assert!(
            error <= tolerance,
            "{}: normwise error {error:e} > {tolerance:e}",
            lines[cursor]
        );
        cursor += consumed;
    }
}

#[test]
fn scipy_savitzky_golay_f64() {
    reference::<f64>();
}
#[test]
fn scipy_savitzky_golay_f32() {
    reference::<f32>();
}

#[test]
fn unsupported_configurations_are_invalid_input() {
    // Degree must be below the window length, derivative order must not exceed
    // degree, and polynomial edge interpolation needs a full window.
    for (window, degree, options) in [
        (5, 5, SavitzkyGolayOptions::<f64>::new()),
        (5, 2, SavitzkyGolayOptions::new().with_derivative(3)),
    ] {
        let error = SavitzkyGolayPlan::new(window, degree, options).unwrap_err();
        assert!(matches!(
            error.view(),
            SciOxideErrorView::InvalidInput { .. }
        ));
    }
    let plan = SavitzkyGolayPlan::new(5, 2, SavitzkyGolayOptions::<f64>::new()).unwrap();
    let error = plan
        .filter(&[1.0, 2.0, 3.0], SavitzkyGolayBoundary::Polynomial)
        .unwrap_err();
    assert!(matches!(
        error.view(),
        SciOxideErrorView::InvalidInput { .. }
    ));
}
