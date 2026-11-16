//! One-shot forms: FIR designs and minimum-phase reconstruction.
use super::{bits, error_kind, invalid};
use scioxide_core::{
    Complex, ExecutionContext, InvalidInputReason, NumericScalar, SciOxideErrorKind, argument,
};
use scioxide_signal::{
    FirBand, FirNormalization, FirSymmetry, MinimumPhaseConvention, MinimumPhaseDiagnostics,
    MinimumPhaseMethod, MinimumPhaseOptions, MinimumPhasePlan, PolyphasePlan, WindowKind,
    design_fir_window, minimum_phase, minimum_phase_with_options,
};

type C = Complex<f64>;

/// Taps and diagnostics of the prepared conversion.
fn prepared<T: NumericScalar>(
    input: &[T],
    options: MinimumPhaseOptions<T::Real>,
) -> (Vec<T>, MinimumPhaseDiagnostics<T::Real>)
where
    Complex<T::Real>: NumericScalar<Real = T::Real>,
{
    let plan = MinimumPhasePlan::<T>::new(input.len(), options).unwrap();
    let mut workspace = plan.workspace(()).unwrap();
    let mut output = vec![T::ZERO; plan.output_len()];
    let cx = ExecutionContext::sequential_native();
    let diagnostics = plan
        .reconstruct_into(input, &mut output, &mut workspace, cx)
        .unwrap();
    (output, diagnostics)
}

fn lowpass() -> Vec<f64> {
    let band = FirBand::new(0.0, 0.3, 1.0, 1.0);
    let (symmetry, normalization) = (FirSymmetry::Even, FirNormalization::None);
    design_fir_window(31, symmetry, &[band], WindowKind::Hamming, normalization)
        .unwrap()
        .coefficients()
        .to_vec()
}

#[test]
fn minimum_phase_equals_the_default_configured_form() {
    let taps = lowpass();
    let (configured, _) = prepared(&taps, MinimumPhaseOptions::default());
    let (with_options, _) =
        minimum_phase_with_options(&taps, MinimumPhaseOptions::default()).unwrap();
    assert_eq!(bits(&minimum_phase(&taps).unwrap()), bits(&configured));
    assert_eq!(bits(&with_options), bits(&configured));
    assert_eq!(configured.len(), 16);
    let complex: Vec<C> = taps
        .iter()
        .enumerate()
        .map(|(k, &h)| C::new(h * (0.4 * k as f64).cos(), h * (0.4 * k as f64).sin()))
        .collect();
    let (configured, _) = prepared(&complex, MinimumPhaseOptions::default());
    assert_eq!(bits(&minimum_phase(&complex).unwrap()), bits(&configured));
    // One tap is its own spectral factor up to the sign of the leading tap.
    let one = minimum_phase(&[-4.0_f64]).unwrap();
    assert_eq!(
        bits(&one),
        bits(&prepared(&[-4.0_f64], MinimumPhaseOptions::default()).0)
    );
}

#[test]
fn minimum_phase_with_options_equals_the_prepared_conversion() {
    let taps = lowpass();
    let options = [
        MinimumPhaseOptions::new().with_convention(MinimumPhaseConvention::SameLength),
        MinimumPhaseOptions::new().with_fft_length(Some(512)),
        MinimumPhaseOptions::new().with_method(MinimumPhaseMethod::Hilbert),
    ];
    for options in options {
        let (owned, diagnostics) = minimum_phase_with_options(&taps, options).unwrap();
        let (expected, reported) = prepared(&taps, options);
        assert_eq!(bits(&owned), bits(&expected));
        assert_eq!(diagnostics, reported);
    }
    let complex: Vec<C> = taps.iter().map(|&h| C::new(h, -0.5 * h)).collect();
    let options = MinimumPhaseOptions::new().with_convention(MinimumPhaseConvention::SameLength);
    let (owned, diagnostics) = minimum_phase_with_options(&complex, options).unwrap();
    let (expected, reported) = prepared(&complex, options);
    assert_eq!(bits(&owned), bits(&expected));
    assert_eq!(diagnostics, reported);
}

#[test]
fn empty_input_is_reported_under_input() {
    let expected = invalid(MinimumPhasePlan::<f64>::new(
        0,
        MinimumPhaseOptions::default(),
    ));
    assert_eq!(expected.0, argument!("input_len"));
    let wanted = (argument!("input"), InvalidInputReason::Empty, expected.2);
    assert_eq!(invalid(minimum_phase::<f64>(&[])), wanted);
    assert_eq!(invalid(minimum_phase::<C>(&[])), wanted);
    assert_eq!(
        invalid(minimum_phase_with_options::<f64>(
            &[],
            MinimumPhaseOptions::new()
        )),
        wanted
    );
    // The empty input is reported before the options are checked.
    let hilbert = MinimumPhaseOptions::new().with_method(MinimumPhaseMethod::Hilbert);
    assert_eq!(
        invalid(minimum_phase_with_options::<f64>(&[], hilbert)),
        wanted
    );
}

#[test]
fn minimum_phase_reports_the_errors_of_the_plan() {
    assert_eq!(
        error_kind(minimum_phase(&[1.0, f64::NAN])),
        SciOxideErrorKind::NonFinite
    );
    assert_eq!(
        error_kind(minimum_phase(&[0.0_f64; 5])),
        SciOxideErrorKind::DomainError
    );
    // A finite input whose output gain is not representable.
    let large = [0.5 * f64::MAX, f64::MAX, -f64::MAX];
    let same = MinimumPhaseOptions::new().with_convention(MinimumPhaseConvention::SameLength);
    assert_eq!(
        error_kind(minimum_phase_with_options(&large, same)),
        SciOxideErrorKind::NumericalInstability
    );
    let hilbert = MinimumPhaseOptions::new().with_method(MinimumPhaseMethod::Hilbert);
    let (_, reason, _) = invalid(minimum_phase_with_options(&[1.0, 1.0], hilbert));
    assert_eq!(reason, InvalidInputReason::UnsupportedConfiguration);
    let short = MinimumPhaseOptions::new().with_fft_length(Some(2));
    let (_, reason, _) = invalid(minimum_phase_with_options(&[1.0, 0.5, 0.25], short));
    assert_eq!(reason, InvalidInputReason::Inconsistent);
}

#[test]
fn to_polyphase_filters_with_the_design_taps() {
    let band = FirBand::new(0.0, 0.3, 1.0, 1.0);
    let (symmetry, normalization) = (FirSymmetry::Even, FirNormalization::None);
    let design = design_fir_window(15, symmetry, &[band], WindowKind::Hann, normalization).unwrap();
    let taps = design.coefficients();
    let x: Vec<f64> = (0..40).map(|i| (0.45 * i as f64).sin()).collect();
    let plan = design.to_polyphase::<f64>().unwrap();
    let expected = PolyphasePlan::fir(taps).unwrap();
    assert_eq!(bits(plan.coefficients()), bits(expected.coefficients()));
    assert_eq!(
        bits(&plan.resample(&x).unwrap()),
        bits(&expected.resample(&x).unwrap())
    );
    let complex: Vec<C> = taps.iter().map(|&h| C::new(h, 0.0)).collect();
    let z: Vec<C> = x.iter().map(|&v| C::new(v, 1.0 - v)).collect();
    let plan = design.to_polyphase::<C>().unwrap();
    let expected = PolyphasePlan::fir(&complex).unwrap();
    assert_eq!(bits(plan.coefficients()), bits(expected.coefficients()));
    assert_eq!(
        bits(&plan.resample(&z).unwrap()),
        bits(&expected.resample(&z).unwrap())
    );
    // The full convolution with the taps applies the same filter.
    let convolved = scioxide_signal::convolve(&x, taps).unwrap();
    let filtered = design.to_polyphase::<f64>().unwrap().resample(&x).unwrap();
    assert_eq!(convolved.len(), filtered.len());
    let error = convolved
        .iter()
        .zip(&filtered)
        .fold(0.0_f64, |e, (a, b)| e.max((a - b).abs()));
    assert!(error < 1e-14, "{error:e}");
}
