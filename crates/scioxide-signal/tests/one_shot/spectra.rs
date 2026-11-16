//! One-shot forms: Welch estimates, Lomb–Scargle periodograms and spectrograms.
use super::{bits, error, error_kind, invalid};
use scioxide_core::{
    AlgebraicScalar, Complex, DomainReason, ExecutionContext, InvalidInputReason, NumericScalar,
    SciOxideError, SciOxideErrorKind, SciOxideErrorView, argument,
};
use scioxide_signal::{
    LombScargle, LombScargleOptions, SpectralEstimateOptions, SpectralEstimatePlan, StftOptions,
    StftPacking, StftPlan, StftRequest, StftScaling, WindowKind, WindowNormalization,
    WindowSymmetry, lomb_scargle, welch, window,
};

type C = Complex<f64>;

fn record(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| (0.3 * i as f64).sin() + 0.4 * (1.7 * i as f64).cos() + 0.2)
        .collect()
}

fn complex_record(n: usize) -> Vec<C> {
    record(n)
        .iter()
        .enumerate()
        .map(|(i, &re)| C::new(re, (0.9 * i as f64).sin()))
        .collect()
}

/// The configured Welch estimate: periodic Hann window, half overlap and the
/// packing of the sample type.
fn configured<T: NumericScalar<Real = f64>>(
    input: &[T],
    sample_rate: f64,
    segment_len: usize,
    packing: StftPacking,
) -> Result<Vec<f64>, SciOxideError> {
    let (symmetry, scale) = (WindowSymmetry::Periodic, WindowNormalization::Native);
    let w = window::<f64>(WindowKind::Hann, segment_len, symmetry, scale)?;
    let options = SpectralEstimateOptions::new()
        .with_sample_rate(sample_rate)
        .with_overlap(segment_len / 2)
        .with_packing(packing);
    SpectralEstimatePlan::<T>::new(&w, options)?.power(input)
}

#[test]
fn welch_equals_the_periodic_hann_half_overlap_estimate() {
    let x = record(1000);
    let fs = 250.0;
    // Acceptance: 64 samples, overlap 32, one-sided.
    let one_sided = configured(&x, fs, 64, StftPacking::OneSided).unwrap();
    assert_eq!(bits(&welch(&x, fs, 64).unwrap()), bits(&one_sided));
    assert_eq!(one_sided.len(), 33);
    for segment_len in [1, 2, 7, 63, 128] {
        let estimate = welch(&x, fs, segment_len).unwrap();
        let expected = configured(&x, fs, segment_len, StftPacking::OneSided).unwrap();
        assert_eq!(
            bits(&estimate),
            bits(&expected),
            "segment_len {segment_len}"
        );
        assert_eq!(estimate.len(), segment_len / 2 + 1);
    }
    // Complex samples take full packing.
    let z = complex_record(1000);
    for segment_len in [1, 64, 63] {
        let estimate = welch(&z, fs, segment_len).unwrap();
        let expected = configured(&z, fs, segment_len, StftPacking::Full).unwrap();
        assert_eq!(
            bits(&estimate),
            bits(&expected),
            "segment_len {segment_len}"
        );
        assert_eq!(estimate.len(), segment_len);
    }
}

#[test]
fn welch_on_one_segment_and_one_sample_short() {
    let x = record(64);
    let single = welch(&x, 2.0, 64).unwrap();
    assert_eq!(
        bits(&single),
        bits(&configured(&x, 2.0, 64, StftPacking::OneSided).unwrap())
    );
    // One segment is the periodogram of the windowed record.
    let (symmetry, scale) = (WindowSymmetry::Periodic, WindowNormalization::Native);
    let w = window::<f64>(WindowKind::Hann, 64, symmetry, scale).unwrap();
    let options = SpectralEstimateOptions::new()
        .with_sample_rate(2.0)
        .with_packing(StftPacking::OneSided);
    let periodogram = SpectralEstimatePlan::<f64>::new(&w, options)
        .unwrap()
        .power(&x)
        .unwrap();
    assert_eq!(bits(&single), bits(&periodogram));
    let short = invalid(welch(&x[..63], 2.0, 64));
    assert_eq!(
        (short.0, short.1),
        (argument!("input"), InvalidInputReason::Inconsistent)
    );
    let z = complex_record(64);
    assert_eq!(welch(&z, 2.0, 64).unwrap().len(), 64);
    assert_eq!(invalid(welch(&z[..63], 2.0, 64)).0, argument!("input"));
}

#[test]
fn welch_names_segment_len_and_sample_rate() {
    let x = record(16);
    // A zero segment length is an empty window.
    let expected = invalid(SpectralEstimatePlan::<f64>::new(
        &[],
        SpectralEstimateOptions::new(),
    ));
    let zero = invalid(welch(&x, 1.0, 0));
    assert_eq!(zero, (argument!("segment_len"), expected.1, expected.2));
    assert_eq!(zero.1, InvalidInputReason::Empty);
    // The rate is checked as the plan checks it, under the parameter name.
    let options = SpectralEstimateOptions::new().with_sample_rate(-1.0);
    let expected = invalid(SpectralEstimatePlan::<f64>::new(&[1.0; 4], options));
    assert_eq!(expected.0, argument!("options.sample_rate"));
    for rate in [0.0, -0.0, -1.0, -f64::MAX] {
        let reported = invalid(welch(&x, rate, 4));
        assert_eq!(
            reported,
            (argument!("sample_rate"), expected.1, expected.2.clone())
        );
        assert_eq!(reported.1, InvalidInputReason::OutOfRange);
        assert_eq!(
            invalid(welch(&complex_record(16), rate, 4)).0,
            argument!("sample_rate")
        );
    }
    // NaN and infinite rates are not finite, including negative infinity.
    for rate in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(error_kind(welch(&x, rate, 4)), SciOxideErrorKind::NonFinite);
    }
    // The segment length is checked before the rate.
    assert_eq!(invalid(welch(&x, -1.0, 0)).0, argument!("segment_len"));
}

#[cfg(target_pointer_width = "64")]
#[test]
fn welch_names_segment_len_beyond_exact_window_coordinates() {
    let x = record(16);
    let limit = 1_usize << 52;
    let (symmetry, scale) = (WindowSymmetry::Periodic, WindowNormalization::Native);
    let expected = invalid(window::<f64>(WindowKind::Hann, limit, symmetry, scale));
    assert_eq!(expected.0, argument!("length"));
    for segment_len in [limit, limit + 1, usize::MAX - 1] {
        assert_eq!(
            invalid(welch(&x, 1.0, segment_len)),
            (argument!("segment_len"), expected.1, expected.2.clone())
        );
    }
    // A length whose sampled form overflows is the window's size overflow.
    assert_eq!(
        error_kind(welch(&x, 1.0, usize::MAX)),
        SciOxideErrorKind::SizeOverflow
    );
}

#[test]
fn welch_reports_samples_and_overflow() {
    let mut x = record(32);
    x[5] = f64::NAN;
    assert_eq!(error_kind(welch(&x, 1.0, 8)), SciOxideErrorKind::NonFinite);
    // Alternating samples near the largest value give an unrepresentable density.
    let large: Vec<f64> = (0..32)
        .map(|i| if i % 2 == 0 { 1e300 } else { -1e300 })
        .collect();
    assert_eq!(
        error_kind(welch(&large, 1.0, 8)),
        SciOxideErrorKind::NumericalInstability
    );
}

#[test]
fn lomb_scargle_equals_the_default_periodogram() {
    let times: Vec<f64> = (0..40)
        .map(|i| 0.13 * i as f64 + 0.05 * (i % 3) as f64)
        .collect();
    let input: Vec<f64> = times.iter().map(|t| 1.5 + (1.1 * t).sin()).collect();
    let frequencies = [0.4, 1.1, 2.3, 5.0];
    let configured = LombScargle::new(&times, &frequencies, None, LombScargleOptions::default())
        .unwrap()
        .power(&input)
        .unwrap();
    let power = lomb_scargle(&times, &input, &frequencies).unwrap();
    assert_eq!(bits(&power), bits(&configured));
    assert!((power[1] - 1.0).abs() < 1e-12);
    // The minimum of three samples for a floating mean.
    let power = lomb_scargle(&times[..3], &input[..3], &frequencies).unwrap();
    assert_eq!(power.len(), 4);
}

#[test]
fn lomb_scargle_reports_the_errors_of_the_periodogram() {
    let times = [0.0, 0.7, 1.9, 2.4];
    let input = [1.0, -0.5, 0.25, 2.0];
    let frequencies = [1.0, 2.0];
    let empty = |result| {
        let (argument, reason, _) = invalid(result);
        assert_eq!(reason, InvalidInputReason::Empty);
        argument
    };
    assert_eq!(
        empty(lomb_scargle(&[], &[], &frequencies)),
        argument!("times")
    );
    assert_eq!(
        empty(lomb_scargle(&times, &input, &[])),
        argument!("frequencies")
    );
    let range = |result| {
        let (argument, reason, _) = invalid(result);
        assert_eq!(reason, InvalidInputReason::OutOfRange);
        argument
    };
    assert_eq!(
        range(lomb_scargle(&times[..2], &input[..2], &frequencies)),
        argument!("times")
    );
    assert_eq!(
        range(lomb_scargle(&times, &input, &[1.0, 0.0])),
        argument!("frequencies")
    );
    assert_eq!(
        error_kind(lomb_scargle(&times, &input[..3], &frequencies)),
        SciOxideErrorKind::DimensionMismatch
    );
    for (t, x) in [
        ([0.0, f64::NAN, 1.9, 2.4], input),
        (times, [1.0, f64::INFINITY, 0.25, 2.0]),
    ] {
        assert_eq!(
            error_kind(lomb_scargle(&t, &x, &frequencies)),
            SciOxideErrorKind::NonFinite
        );
    }
    assert!(matches!(
        error(lomb_scargle(&times, &[3.0; 4], &frequencies)).view(),
        SciOxideErrorView::DomainError { argument, reason: DomainReason::Undefined, .. }
            if argument == argument!("input")
    ));
    // The phase ω (tᵢ − t₀) overflows.
    let spread = [0.0, 1e10, 2e10, 3e10];
    assert_eq!(
        error_kind(lomb_scargle(&spread, &input, &[1e300])),
        SciOxideErrorKind::NumericalInstability
    );
}

fn spectrogram_into<T: NumericScalar>(plan: &StftPlan<T>, input: &[T]) -> Vec<T::Real>
where
    Complex<T::Real>: NumericScalar<Real = T::Real>,
{
    let mut output = vec![T::Real::ZERO; plan.output_len(input.len()).unwrap()];
    let mut ws = plan
        .workspace(StftRequest::Spectrogram(input.len()))
        .unwrap();
    let cx = ExecutionContext::sequential_native();
    plan.spectrogram_into(input, &mut output, &mut ws, cx)
        .unwrap();
    output
}

#[test]
fn spectrogram_equals_the_prepared_form() {
    let (symmetry, scale) = (WindowSymmetry::Periodic, WindowNormalization::Native);
    let w = window::<f64>(WindowKind::Hann, 16, symmetry, scale).unwrap();
    let options = StftOptions::new()
        .with_packing(StftPacking::OneSided)
        .with_scaling(StftScaling::Psd)
        .with_sample_rate(4.0);
    let plan = StftPlan::<f64>::new(&w, 4, options).unwrap();
    for n in [0, 1, 15, 100] {
        let x = record(n);
        let power = plan.spectrogram(&x).unwrap();
        assert_eq!(power.len(), plan.output_len(n).unwrap());
        assert_eq!(bits(&power), bits(&spectrogram_into(&plan, &x)));
    }
    let complex: Vec<C> = w.iter().map(|&v| C::new(v, 0.0)).collect();
    let plan = StftPlan::<C>::new(&complex, 5, StftOptions::new()).unwrap();
    let z = complex_record(77);
    assert_eq!(
        bits(&plan.spectrogram(&z).unwrap()),
        bits(&spectrogram_into(&plan, &z))
    );
}

#[test]
fn spectrogram_reports_samples_and_overflow() {
    let plan = StftPlan::<f64>::new(&[1.0; 4], 2, StftOptions::new()).unwrap();
    assert_eq!(
        error_kind(plan.spectrogram(&[0.0, f64::NAN, 1.0])),
        SciOxideErrorKind::NonFinite
    );
    assert_eq!(
        error_kind(plan.spectrogram(&[f64::MAX; 8])),
        SciOxideErrorKind::NumericalInstability
    );
}
