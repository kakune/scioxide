//! Exact minimum-phase factors and SciPy `minimum_phase` references.
//!
//! Fixture: `fixtures/minimum_phase_scipy.tsv`, written by
//! `generate_minimum_phase.py`. Inputs are
//! built from exact zeros, so every record stores the exact factor (50-digit
//! mpmath expansion); cepstral aliasing at the stored FFT length is below
//! 1e-17. Real homomorphic records also store SciPy's output, which differs
//! from the exact factor by its additive `1e-7 * min |H|` log floor; that
//! stored distance bounds the SciOxide-SciPy comparison. Errors are normwise:
//! the largest coefficient error divided by `max(max |reference|, 1)`.
#![allow(clippy::unwrap_used)]
use scioxide_core::{Complex, ExecutionContext, NumericScalar, RealScalar, SciOxideErrorView};
use scioxide_signal::*;

fn numbers(line: &str) -> Vec<f64> {
    line.split_whitespace()
        .map(|x| x.parse().unwrap())
        .collect()
}
fn complexes(line: &str) -> Vec<Complex<f64>> {
    numbers(line)
        .chunks_exact(2)
        .map(|v| Complex::new(v[0], v[1]))
        .collect()
}
fn normwise<T: NumericScalar>(actual: &[T], expected: &[Complex<f64>]) -> f64 {
    assert_eq!(actual.len(), expected.len());
    let scale = expected.iter().fold(1.0_f64, |m, x| m.max(x.norm()));
    actual
        .iter()
        .zip(expected)
        .map(|(a, &b)| {
            (Complex::new(
                a.real_component().into_f64(),
                a.imaginary_component().into_f64(),
            ) - b)
                .norm()
        })
        .fold(0.0, f64::max)
        / scale
}

fn reconstruct<T: NumericScalar>(case: &[&str], tolerance: f64)
where
    Complex<T::Real>: NumericScalar<Real = T::Real>,
{
    let header: Vec<_> = case[0].split_whitespace().collect();
    let method = match header[0] {
        "homomorphic" => MinimumPhaseMethod::Homomorphic,
        _ => MinimumPhaseMethod::Hilbert,
    };
    let convention = match header[1] {
        "same" => MinimumPhaseConvention::SameLength,
        _ => MinimumPhaseConvention::HalfLength,
    };
    let n: usize = header[3].parse().unwrap();
    let fft_length: usize = header[4].parse().unwrap();
    let input: Vec<T> = complexes(case[1])
        .into_iter()
        .map(|z| {
            T::from_components(
                T::Real::try_from_f64_lossy(z.re).unwrap(),
                T::Real::try_from_f64_lossy(z.im).unwrap(),
            )
        })
        .collect();
    assert_eq!(input.len(), n);
    let options = MinimumPhaseOptions::new()
        .with_method(method)
        .with_convention(convention)
        .with_fft_length(Some(fft_length));
    let plan = MinimumPhasePlan::<T>::new(n, options).unwrap();
    let mut workspace = plan.workspace(()).unwrap();
    let mut output = vec![T::ZERO; plan.output_len()];
    let diagnostics = plan
        .reconstruct_into(
            &input,
            &mut output,
            &mut workspace,
            ExecutionContext::sequential_native(),
        )
        .unwrap();
    assert_eq!(diagnostics.fft_length(), fft_length);
    assert_eq!(diagnostics.regularized_bins(), 0);
    if method == MinimumPhaseMethod::Hilbert {
        // A strictly positive zero-phase amplitude needs no lift.
        assert_eq!(diagnostics.hilbert_lift().into_f64(), 0.0);
    }
    let (convenience, _) = minimum_phase_with_options(&input, options).unwrap();
    assert!(convenience == output, "{}", case[0]);
    let error = normwise(&output, &complexes(case[2]));
    assert!(
        error <= tolerance,
        "{}: exact error {error:e} > {tolerance:e}",
        case[0]
    );
    if case[3] != "-" {
        let scipy: Vec<_> = numbers(case[3])
            .into_iter()
            .map(|x| Complex::new(x, 0.0))
            .collect();
        let distance: f64 = case[4].parse().unwrap();
        let error = normwise(&output, &scipy);
        assert!(
            error <= distance + tolerance,
            "{}: SciPy error {error:e} > {distance:e} + {tolerance:e}",
            case[0]
        );
    }
}

/// f64 meets 1e-12 (observed worst 3.2e-16). Inputs are binary fractions
/// exact in f32, so f32 errors are the rounding of the cepstral FFT pipeline
/// alone: observed at most 2.5 epsilon, bounded here by 64 epsilon.
fn reference<S>()
where
    S: RealScalar + NumericScalar<Real = S>,
    Complex<S>: NumericScalar<Real = S>,
{
    let tolerance = if S::MANTISSA_DIGITS == 24 {
        64.0 * S::EPSILON.into_f64()
    } else {
        1e-12
    };
    let lines: Vec<_> = include_str!("fixtures/minimum_phase_scipy.tsv")
        .lines()
        .filter(|s| !s.starts_with('#'))
        .collect();
    for case in lines.chunks_exact(5) {
        if case[0].split_whitespace().nth(2) == Some("1") {
            reconstruct::<Complex<S>>(case, tolerance);
        } else {
            reconstruct::<S>(case, tolerance);
        }
    }
}

#[test]
fn exact_and_scipy_minimum_phase_f64() {
    reference::<f64>();
}
#[test]
fn exact_and_scipy_minimum_phase_f32() {
    reference::<f32>();
}

#[test]
fn unsupported_configurations_are_invalid_input() {
    // Same-length output requires the homomorphic method, and the FFT length
    // must accommodate the input.
    for options in [
        MinimumPhaseOptions::<f64>::new()
            .with_method(MinimumPhaseMethod::Hilbert)
            .with_convention(MinimumPhaseConvention::SameLength),
        MinimumPhaseOptions::new().with_fft_length(Some(4)),
    ] {
        let error = MinimumPhasePlan::<f64>::new(5, options).unwrap_err();
        assert!(matches!(
            error.view(),
            SciOxideErrorView::InvalidInput { .. }
        ));
    }
}
