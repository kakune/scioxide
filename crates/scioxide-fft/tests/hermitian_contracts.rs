//! Packed real and Hermitian storage, public forms and transactional failures.
#![expect(clippy::unwrap_used)]
use scioxide_core::{Complex, ExecutionContext, SciOxideErrorKind};
use scioxide_fft::{
    FftDirection, FftInput, FftLayout, FftOutput, FftPlan, FftPlanSpec, FftWorkspace, Normalization,
};
use scioxide_linalg::{VectorView, VectorViewMut};

fn plan(n: usize, direction: FftDirection, layout: FftLayout) -> FftPlan<f64> {
    FftPlan::build(
        FftPlanSpec::new(n, direction, Normalization::Inverse, layout),
        ExecutionContext::sequential(),
    )
    .unwrap()
}

#[test]
fn strided_real_and_hermitian_storage_preserves_gaps_and_logical_lengths() {
    for n in [5, 6] {
        let forward = plan(n, FftDirection::Forward, FftLayout::Real);
        let inverse = plan(n, FftDirection::Inverse, FftLayout::Real);
        let input: Vec<_> = (0..n * 2)
            .map(|j| {
                if j % 2 == 0 {
                    (j as f64 * 0.3).sin()
                } else {
                    91.0
                }
            })
            .collect();
        let sentinel = Complex::new(73.0, -19.0);
        let bins = n / 2 + 1;
        let mut spectrum = vec![sentinel; bins * 3];
        let mut work = forward.workspace(()).unwrap();
        work.try_reserve(&inverse.workspace_requirements(()).unwrap())
            .unwrap();
        forward
            .execute_into(
                FftInput::Real(VectorView::from_slice_with_stride(&input, n, 2).unwrap()),
                FftOutput::Hermitian {
                    logical_len: n,
                    bins: VectorViewMut::from_slice_with_stride(&mut spectrum, bins, 3).unwrap(),
                },
                &mut work,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert!(
            spectrum
                .iter()
                .enumerate()
                .all(|(j, value)| j % 3 == 0 || *value == sentinel)
        );
        let mut restored = vec![83.0; n * 2];
        inverse
            .execute_into(
                FftInput::Hermitian {
                    logical_len: n,
                    bins: VectorView::from_slice_with_stride(&spectrum, bins, 3).unwrap(),
                },
                FftOutput::Real(
                    VectorViewMut::from_slice_with_stride(&mut restored, n, 2).unwrap(),
                ),
                &mut work,
                ExecutionContext::sequential(),
            )
            .unwrap();
        for j in 0..n {
            assert!((restored[j * 2] - input[j * 2]).abs() < 1e-12);
            assert_eq!(restored[j * 2 + 1], 83.0);
        }
        let before = restored.clone();
        let error = inverse
            .execute_into(
                FftInput::Hermitian {
                    logical_len: n ^ 1,
                    bins: VectorView::from_slice_with_stride(&spectrum, bins, 3).unwrap(),
                },
                FftOutput::Real(
                    VectorViewMut::from_slice_with_stride(&mut restored, n, 2).unwrap(),
                ),
                &mut work,
                ExecutionContext::sequential(),
            )
            .unwrap_err();
        assert_eq!(error.kind(), SciOxideErrorKind::DimensionMismatch);
        assert_eq!(restored, before);
    }
}

#[test]
fn strided_and_contiguous_packed_views_are_bitwise_identical() {
    for n in [1, 2, 5, 6, 63, 64] {
        let bins = n / 2 + 1;
        // `ihfft`/`hfft` run the opposite real kernel with conjugated bins.
        for (layout, to_bins, to_samples) in [
            (
                FftLayout::Real,
                FftDirection::Forward,
                FftDirection::Inverse,
            ),
            (
                FftLayout::Hermitian,
                FftDirection::Inverse,
                FftDirection::Forward,
            ),
        ] {
            let forward = plan(n, to_bins, layout);
            let inverse = plan(n, to_samples, layout);
            let mut work = forward.workspace(()).unwrap();
            work.try_reserve(&inverse.workspace_requirements(()).unwrap())
                .unwrap();
            let samples: Vec<_> = (0..n)
                .map(|j| (j as f64 * 0.7).sin() * 2.0_f64.powi(j as i32 % 5 - 2))
                .collect();
            let strided_samples: Vec<_> = samples.iter().flat_map(|&x| [x, -91.0]).collect();

            let mut contiguous = vec![Complex::new(0.0, 0.0); bins];
            let mut strided = vec![Complex::new(73.0, -19.0); bins * 3];
            for (input, output) in [
                (
                    VectorView::from_slice(&samples),
                    VectorViewMut::from_slice(&mut contiguous),
                ),
                (
                    VectorView::from_slice_with_stride(&strided_samples, n, 2).unwrap(),
                    VectorViewMut::from_slice_with_stride(&mut strided, bins, 3).unwrap(),
                ),
            ] {
                forward
                    .execute_into(
                        FftInput::Real(input),
                        FftOutput::Hermitian {
                            logical_len: n,
                            bins: output,
                        },
                        &mut work,
                        ExecutionContext::sequential(),
                    )
                    .unwrap();
            }
            let strided_bins: Vec<_> = strided.iter().step_by(3).copied().collect();
            assert_eq!(bits(&strided_bins), bits(&contiguous), "{layout:?} n={n}");

            let mut restored = vec![0.0; n];
            let mut strided_restored = vec![83.0; n * 2];
            for (input, output) in [
                (
                    VectorView::from_slice(&contiguous),
                    VectorViewMut::from_slice(&mut restored),
                ),
                (
                    VectorView::from_slice_with_stride(&strided, bins, 3).unwrap(),
                    VectorViewMut::from_slice_with_stride(&mut strided_restored, n, 2).unwrap(),
                ),
            ] {
                inverse
                    .execute_into(
                        FftInput::Hermitian {
                            logical_len: n,
                            bins: input,
                        },
                        FftOutput::Real(output),
                        &mut work,
                        ExecutionContext::sequential(),
                    )
                    .unwrap();
            }
            let strided_restored: Vec<_> = strided_restored.iter().step_by(2).copied().collect();
            assert_eq!(
                strided_restored
                    .iter()
                    .map(|x| x.to_bits())
                    .collect::<Vec<_>>(),
                restored.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                "{layout:?} n={n}"
            );
            for (actual, expected) in restored.iter().zip(&samples) {
                assert!((actual - expected).abs() < 1e-13, "{layout:?} n={n}");
            }
        }
    }
}

fn bits(values: &[Complex<f64>]) -> Vec<(u64, u64)> {
    values
        .iter()
        .map(|z| (z.re.to_bits(), z.im.to_bits()))
        .collect()
}

#[test]
fn hermitian_plan_failures_leave_output_unchanged() {
    let build = |n, direction| {
        FftPlan::<f64>::build(
            FftPlanSpec::new(n, direction, Normalization::Inverse, FftLayout::Hermitian),
            ExecutionContext::sequential(),
        )
        .unwrap()
    };
    let forward = build(6, FftDirection::Forward);
    let mut scratch = forward.workspace(()).unwrap();
    let hfft_kind = |bins: &[Complex<f64>], logical_len, scratch: &mut FftWorkspace<f64>| {
        let mut output = [9.0; 6];
        let error = forward
            .execute_into(
                FftInput::Hermitian {
                    logical_len,
                    bins: VectorView::from_slice(bins),
                },
                &mut output,
                scratch,
                ExecutionContext::sequential(),
            )
            .unwrap_err();
        assert_eq!(output, [9.0; 6]);
        error.kind()
    };
    let valid = [Complex::new(1.0, 0.0); 4];
    // Asymmetric DC and Nyquist endpoints violate the Hermitian input policy.
    for endpoint in [0, 3] {
        let mut bins = valid;
        bins[endpoint].im = 0.5;
        assert_eq!(
            hfft_kind(&bins, 6, &mut scratch),
            SciOxideErrorKind::InvalidInput
        );
    }
    let mut bins = valid;
    bins[1].re = f64::NAN;
    assert_eq!(
        hfft_kind(&bins, 6, &mut scratch),
        SciOxideErrorKind::NonFinite
    );
    // n = 6 and n = 7 share four packed bins; the logical parity must match.
    assert_eq!(
        hfft_kind(&valid, 7, &mut scratch),
        SciOxideErrorKind::DimensionMismatch
    );

    let inverse = build(7, FftDirection::Inverse);
    let mut scratch = inverse.workspace(()).unwrap();
    let mut packed = [Complex::new(9.0, 9.0); 4];
    let mut samples = [1.0; 7];
    samples[3] = f64::INFINITY;
    for (input, logical_len, kind) in [
        (samples, 7, SciOxideErrorKind::NonFinite),
        ([1.0; 7], 6, SciOxideErrorKind::DimensionMismatch),
    ] {
        let error = inverse
            .execute_into(
                FftInput::from(&input),
                FftOutput::Hermitian {
                    logical_len,
                    bins: VectorViewMut::from_slice(&mut packed),
                },
                &mut scratch,
                ExecutionContext::sequential(),
            )
            .unwrap_err();
        assert_eq!(error.kind(), kind);
        assert_eq!(packed, [Complex::new(9.0, 9.0); 4]);
    }
    // A Hermitian inverse consumes real samples; the real-layout pairing is rejected.
    let mut real = [9.0; 7];
    let error = inverse
        .execute_into(
            FftInput::Hermitian {
                logical_len: 7,
                bins: VectorView::from_slice(&valid),
            },
            &mut real,
            &mut scratch,
            ExecutionContext::sequential(),
        )
        .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::InvalidInput);
    assert_eq!(real, [9.0; 7]);
}
