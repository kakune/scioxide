//! SciPy `firls`, `firwin` and `firwin2` references for the direct FIR designs.
//!
//! Fixture: `fixtures/fir_scipy.tsv`, written by
//! `generate_fir.py` (convention mapping in
//! its docstring). Errors are normwise: the largest tap error divided by
//! `max(max |reference tap|, 1)`.
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
fn window(name: &str, beta: f64) -> WindowKind<'static> {
    match name {
        "hamming" => WindowKind::Hamming,
        "kaiser" => WindowKind::Kaiser { beta },
        _ => {
            assert_eq!(name, "boxcar");
            WindowKind::Boxcar
        }
    }
}
fn normwise(actual: &[impl RealScalar], expected: &[f64]) -> f64 {
    assert_eq!(actual.len(), expected.len());
    let scale = expected.iter().fold(1.0_f64, |m, x| m.max(x.abs()));
    actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a.into_f64() - b).abs())
        .fold(0.0, f64::max)
        / scale
}

/// f64 meets 1e-12 normwise (observed worst 1.4e-14, firls with 31 taps). In
/// f32 the band edges, weights and knots are themselves rounded (half an
/// epsilon each) and the least-squares normal equations amplify that
/// perturbation; 256 epsilon covers the observed worst case (about 47 epsilon,
/// firls with 101 taps).
fn reference<S: RealScalar>() {
    let tolerance = if S::MANTISSA_DIGITS == 24 {
        256.0 * S::EPSILON.into_f64()
    } else {
        1e-12
    };
    let lines: Vec<_> = include_str!("fixtures/fir_scipy.tsv")
        .lines()
        .filter(|s| !s.starts_with('#'))
        .collect();
    for case in lines.chunks_exact(3) {
        let header: Vec<_> = case[0].split_whitespace().collect();
        let n: usize = header[1].parse().unwrap();
        let symmetry = if header[2] == "1" {
            FirSymmetry::Odd
        } else {
            FirSymmetry::Even
        };
        let beta: f64 = header[4].parse().unwrap();
        let at: f64 = header[5].parse().unwrap();
        let normalization = if at.is_nan() {
            FirNormalization::None
        } else {
            FirNormalization::At {
                frequency: s(at),
                gain: S::ONE,
            }
        };
        let spec = numbers(case[1]);
        let bands: Vec<_> = spec
            .chunks_exact(5)
            .map(|b| FirBand::new(s(b[0]), s(b[1]), s(b[2]), s(b[3])).with_weight(s(b[4])))
            .collect();
        let design = match header[0] {
            "firls" => design_fir_least_squares::<S>(n, symmetry, &bands),
            "firwin" => {
                design_fir_window(n, symmetry, &bands, window(header[3], beta), normalization)
            }
            kind => {
                assert_eq!(kind, "firwin2");
                let (knots, gains) = spec.split_at(spec.len() / 2);
                let knots: Vec<S> = knots.iter().map(|&v| s(v)).collect();
                let gains: Vec<S> = gains.iter().map(|&v| s(v)).collect();
                design_fir_frequency_sampling(
                    n,
                    symmetry,
                    &knots,
                    &gains,
                    header[6].parse().unwrap(),
                    (header[3] != "boxcar").then(|| window(header[3], beta)),
                    normalization,
                )
            }
        }
        .unwrap();
        let error = normwise(design.coefficients(), &numbers(case[2]));
        assert!(
            error <= tolerance,
            "{} n={n}: normwise error {error:e} > {tolerance:e}",
            header[0]
        );
    }
}

#[test]
fn scipy_fir_designs_f64() {
    reference::<f64>();
}
#[test]
fn scipy_fir_designs_f32() {
    reference::<f32>();
}

#[test]
fn unsupported_configurations_are_invalid_input() {
    // A Type II design has a forced zero at Nyquist, so a nonzero requested
    // gain there is invalid input.
    let error = design_fir_frequency_sampling(
        32,
        FirSymmetry::Even,
        &[0.0_f64, 1.0],
        &[1.0, 1.0],
        64,
        None,
        FirNormalization::None,
    )
    .unwrap_err();
    assert!(matches!(
        error.view(),
        SciOxideErrorView::InvalidInput { .. }
    ));
    // A mesh shorter than twice the tap count.
    let error = design_fir_frequency_sampling(
        31,
        FirSymmetry::Even,
        &[0.0_f64, 1.0],
        &[1.0, 0.0],
        32,
        None,
        FirNormalization::None,
    )
    .unwrap_err();
    assert!(matches!(
        error.view(),
        SciOxideErrorView::InvalidInput { .. }
    ));
}
